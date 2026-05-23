from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from task_featgen_config import TASK_FEATGEN_CONFIGS

_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_DEFAULT_TASK = "entity_matching"
_TASK_TO_SCOPE = {
    str(task_name): str(cfg.get("task_scope", "row_pair"))
    for task_name, cfg in TASK_FEATGEN_CONFIGS.items()
}


class GeneratedFeatureSpecError(ValueError):
    """Raised when a generated feature spec is invalid."""


class GeneratedFeatureExecutionError(RuntimeError):
    """Raised when a generated feature function fails at runtime."""


@dataclass(frozen=True)
class GeneratedFeatureSpec:
    feature_name: str
    task: str
    scope: str
    version: str
    description: str
    inputs_used: Tuple[str, ...]
    code: str
    fallback_value: float
    range_hint: Optional[Tuple[float, float]]
    example_based_on: Tuple[str, ...]
    code_hash: str

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "feature_name": self.feature_name,
            "task": self.task,
            "scope": self.scope,
            "version": self.version,
            "description": self.description,
            "inputs_used": list(self.inputs_used),
            "code": self.code,
            "fallback_value": float(self.fallback_value),
            "example_based_on": list(self.example_based_on),
            "code_hash": self.code_hash,
        }
        if self.range_hint is not None:
            payload["range_hint"] = [float(self.range_hint[0]), float(self.range_hint[1])]
        return payload


_ALLOWED_AST_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.AugAssign,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.ListComp,
    ast.comprehension,
    ast.Expr,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
}

_ALLOWED_BUILTIN_CALLS = {
    "abs",
    "min",
    "max",
    "range",
    "float",
    "int",
    "len",
    "sum",
    "sorted",
    "bool",
    "list",
    "set",
    "tuple",
    "str",
    "enumerate",
}
_ALLOWED_METHOD_CALLS = {"add", "append", "get", "keys", "intersection", "union"}


def _normalize_name(value: Any, *, field: str) -> str:
    token = str(value).strip()
    if not token:
        raise GeneratedFeatureSpecError(f"{field} is required.")
    return token


def _normalize_range_hint(raw: Any) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise GeneratedFeatureSpecError("range_hint must be [lo, hi].")
    lo = float(raw[0])
    hi = float(raw[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise GeneratedFeatureSpecError("range_hint values must be finite.")
    if lo >= hi:
        raise GeneratedFeatureSpecError("range_hint must satisfy lo < hi.")
    return (lo, hi)


def _dedup_strs(values: Sequence[Any]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for item in values:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _call_target_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return str(node.id)
    if isinstance(node, ast.Attribute):
        if str(getattr(node, "attr", "")).strip() in _ALLOWED_METHOD_CALLS:
            return f"method:{str(node.attr).strip()}"
        return None
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and str(base.id) == "helpers":
            return "helpers[]"
        if isinstance(base, ast.Subscript):
            root = base.value
            if isinstance(root, ast.Name) and str(root.id) == "ctx":
                return "helpers[]"
    return None


def _is_helpers_container_expr(node: ast.AST, helper_aliases: Optional[set[str]] = None) -> bool:
    if isinstance(node, ast.Name):
        token = str(node.id)
        return token == "helpers" or (helper_aliases is not None and token in helper_aliases)
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and str(base.id) == "ctx":
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and str(slice_node.value) == "helpers":
                return True
    return False


class _GeneratedFeatureValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_defs: List[ast.FunctionDef] = []
        self.helper_aliases: set[str] = set()

    def generic_visit(self, node: ast.AST):
        if type(node) not in _ALLOWED_AST_NODES:
            raise GeneratedFeatureSpecError(f"Unsupported syntax node in generated code: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_defs.append(node)
        if node.name != "compute_feature":
            raise GeneratedFeatureSpecError("Generated code must define exactly one function named compute_feature.")
        if len(node.args.args) != 1 or str(node.args.args[0].arg) != "ctx":
            raise GeneratedFeatureSpecError("compute_feature must take exactly one argument: ctx")
        if node.decorator_list:
            raise GeneratedFeatureSpecError("Decorators are not allowed in generated feature code.")
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _is_helpers_container_expr(node.value, self.helper_aliases) or _call_target_name(node.value) == "helpers[]":
                self.helper_aliases.add(str(node.targets[0].id))
        for target in node.targets:
            self.visit(target)
        self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute):
        attr = str(getattr(node, "attr", "")).strip()
        if not attr or attr.startswith("__"):
            raise GeneratedFeatureSpecError("Dunder/private attribute access is not allowed.")
        if attr not in _ALLOWED_METHOD_CALLS:
            raise GeneratedFeatureSpecError(
                f"Attribute access {attr!r} is not allowed; methods={sorted(_ALLOWED_METHOD_CALLS)}"
            )
        self.visit(node.value)

    def visit_Call(self, node: ast.Call):
        if node.keywords:
            raise GeneratedFeatureSpecError("Keyword arguments are not allowed in generated feature code.")
        target_name = _call_target_name(node.func)
        if isinstance(node.func, ast.Name) and str(node.func.id) in self.helper_aliases:
            target_name = "helper_alias"
        if isinstance(node.func, ast.Subscript) and _is_helpers_container_expr(node.func.value, self.helper_aliases):
            target_name = "helpers[]"
        if target_name is None:
            raise GeneratedFeatureSpecError("Only direct builtin calls or helpers[...] calls are allowed.")
        if (
            target_name != "helpers[]"
            and target_name != "helper_alias"
            and target_name not in _ALLOWED_BUILTIN_CALLS
            and target_name not in {f"method:{name}" for name in _ALLOWED_METHOD_CALLS}
        ):
            raise GeneratedFeatureSpecError(
                f"Call target {target_name!r} is not allowed; builtins={sorted(_ALLOWED_BUILTIN_CALLS)}"
            )
        for arg in node.args:
            self.visit(arg)


def _validate_generated_code(code: str) -> None:
    source = str(code).strip()
    if not source:
        raise GeneratedFeatureSpecError("code is required.")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise GeneratedFeatureSpecError(f"Invalid generated feature code: {exc}") from exc
    validator = _GeneratedFeatureValidator()
    validator.visit(tree)
    if len(validator.function_defs) != 1:
        raise GeneratedFeatureSpecError("Generated code must contain exactly one function definition.")


def validate_generated_feature_spec(
    doc: Mapping[str, Any],
    *,
    expected_task: str = _DEFAULT_TASK,
    expected_scope: str = "",
) -> GeneratedFeatureSpec:
    if not isinstance(doc, Mapping):
        raise GeneratedFeatureSpecError("Generated feature spec must be a JSON object.")

    feature_name = _normalize_name(doc.get("feature_name", ""), field="feature_name")
    if not _FEATURE_NAME_RE.match(feature_name):
        raise GeneratedFeatureSpecError(
            "feature_name must match ^[a-z][a-z0-9_]{1,63}$ "
            f"(got {feature_name!r})"
        )

    task = _normalize_name(doc.get("task", expected_task), field="task")
    if task != str(expected_task):
        raise GeneratedFeatureSpecError(f"task mismatch: expected {expected_task!r}, got {task!r}")

    scope_expected = str(expected_scope).strip() or str(_TASK_TO_SCOPE.get(str(expected_task), "row_pair"))
    scope = _normalize_name(doc.get("scope", scope_expected), field="scope")
    if scope != scope_expected:
        raise GeneratedFeatureSpecError(f"scope must be {scope_expected!r}, got {scope!r}")

    version = _normalize_name(doc.get("version", "v1"), field="version")
    description = str(doc.get("description", "")).strip()
    inputs_used_raw = doc.get("inputs_used", [])
    if inputs_used_raw is None:
        inputs_used_raw = []
    if not isinstance(inputs_used_raw, list):
        raise GeneratedFeatureSpecError("inputs_used must be a list.")
    inputs_used = _dedup_strs(inputs_used_raw)

    code = str(doc.get("code", "")).strip()
    _validate_generated_code(code)

    fallback_value = float(doc.get("fallback_value", 0.0))
    if not math.isfinite(fallback_value):
        raise GeneratedFeatureSpecError("fallback_value must be finite.")

    range_hint = _normalize_range_hint(doc.get("range_hint", None))
    example_based_on_raw = doc.get("example_based_on", [])
    if example_based_on_raw is None:
        example_based_on_raw = []
    if isinstance(example_based_on_raw, str):
        example_based_on_raw = [example_based_on_raw]
    if not isinstance(example_based_on_raw, list):
        raise GeneratedFeatureSpecError("example_based_on must be a list.")
    example_based_on = _dedup_strs(example_based_on_raw)

    canonical = {
        "feature_name": feature_name,
        "task": task,
        "scope": scope,
        "version": version,
        "description": description,
        "inputs_used": list(inputs_used),
        "code": code,
        "fallback_value": float(fallback_value),
        "range_hint": list(range_hint) if range_hint is not None else None,
        "example_based_on": list(example_based_on),
    }
    code_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return GeneratedFeatureSpec(
        feature_name=feature_name,
        task=task,
        scope=scope,
        version=version,
        description=description,
        inputs_used=inputs_used,
        code=code,
        fallback_value=float(fallback_value),
        range_hint=range_hint,
        example_based_on=example_based_on,
        code_hash=code_hash,
    )


def _load_feature_docs_from_file(path: Path) -> List[Mapping[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if "features" in raw:
            features = raw.get("features", [])
            if not isinstance(features, list):
                raise GeneratedFeatureSpecError(f"'features' must be a list in {path}")
            return [item for item in features if isinstance(item, Mapping)]
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    raise GeneratedFeatureSpecError(f"Unsupported generated feature JSON format: {path}")


def load_generated_feature_specs(
    path: str | Path,
    *,
    expected_task: str = _DEFAULT_TASK,
    expected_scope: str = "",
) -> List[GeneratedFeatureSpec]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Generated feature path not found: {src}")

    docs: List[Mapping[str, Any]] = []
    if src.is_dir():
        for fp in sorted(src.glob("*.json")):
            docs.extend(_load_feature_docs_from_file(fp))
    else:
        docs.extend(_load_feature_docs_from_file(src))

    specs: List[GeneratedFeatureSpec] = []
    seen = set()
    for doc in docs:
        spec = validate_generated_feature_spec(
            doc,
            expected_task=expected_task,
            expected_scope=expected_scope,
        )
        if spec.feature_name in seen:
            raise GeneratedFeatureSpecError(f"Duplicate generated feature name: {spec.feature_name}")
        seen.add(spec.feature_name)
        specs.append(spec)
    return specs


def save_generated_feature_specs(
    specs: Sequence[GeneratedFeatureSpec],
    path: str | Path,
    *,
    task: str = _DEFAULT_TASK,
    scope: str = "",
) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    resolved_scope = str(scope).strip() or str(_TASK_TO_SCOPE.get(str(task), "row_pair"))
    payload = {
        "task": str(task),
        "scope": str(resolved_scope),
        "version": "v1",
        "features": [spec.to_dict() for spec in specs],
    }
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compile_generated_feature(spec: GeneratedFeatureSpec) -> Callable[[Mapping[str, Any]], float]:
    local_env: Dict[str, Any] = {}
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "float": float,
        "int": int,
        "len": len,
        "sum": sum,
        "sorted": sorted,
        "bool": bool,
        "list": list,
        "set": set,
        "tuple": tuple,
        "str": str,
    }
    exec_globals = {"__builtins__": allowed_builtins}
    exec(spec.code, exec_globals, local_env)
    fn = local_env.get("compute_feature", None)
    if not callable(fn):
        raise GeneratedFeatureSpecError(f"Generated feature {spec.feature_name!r} did not define callable compute_feature.")

    def _wrapped(ctx: Mapping[str, Any]) -> float:
        try:
            value = fn(ctx)
            value_f = float(value)
            if not math.isfinite(value_f):
                raise GeneratedFeatureExecutionError(
                    f"Generated feature {spec.feature_name!r} returned non-finite value: {value!r}"
                )
            if spec.range_hint is not None:
                lo, hi = spec.range_hint
                if value_f < lo:
                    value_f = float(lo)
                elif value_f > hi:
                    value_f = float(hi)
            return float(value_f)
        except Exception as exc:
            raise GeneratedFeatureExecutionError(
                f"Generated feature {spec.feature_name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    return _wrapped


class GeneratedFeatureRegistry:
    def __init__(self, specs: Sequence[GeneratedFeatureSpec]) -> None:
        self.specs: Tuple[GeneratedFeatureSpec, ...] = tuple(specs)
        self.feature_names: Tuple[str, ...] = tuple(spec.feature_name for spec in self.specs)
        self.compiled: Dict[str, Callable[[Mapping[str, Any]], float]] = {
            spec.feature_name: _compile_generated_feature(spec) for spec in self.specs
        }
        self.fingerprint = hashlib.sha1(
            json.dumps(
                [{"feature_name": spec.feature_name, "code_hash": spec.code_hash} for spec in self.specs],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

    def compute(self, ctx: Mapping[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self.specs:
            fn = self.compiled[spec.feature_name]
            try:
                out[spec.feature_name] = float(fn(ctx))
            except Exception:
                out[spec.feature_name] = float(spec.fallback_value)
        return out


def load_generated_feature_registry(
    path: str | Path,
    *,
    expected_task: str = _DEFAULT_TASK,
    expected_scope: str = "",
) -> GeneratedFeatureRegistry:
    specs = load_generated_feature_specs(
        path,
        expected_task=expected_task,
        expected_scope=expected_scope,
    )
    return GeneratedFeatureRegistry(specs)
