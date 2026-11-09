from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


SPEC_VERSION = "v1"
SUPPORTED_SPEC_VERSIONS = {"v1", "v2"}
_MIN_V2_CHANNELS = 1
_MAX_V2_CHANNELS = 32
_DECISION_OPS = {">=", ">", "<=", "<"}
_AGGREGATION_METHODS = {"weighted_sum"}
_POSTPROCESS_OPS = {"none", "sigmoid", "clip01"}
_ALLOW_DUP_SIGNATURE = str(os.getenv("SYMBOLIC_ALLOW_DUP_SIGNATURE", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
}


class SymbolicSpecError(ValueError):
    """Raised when a symbolic spec is invalid."""


class SymbolicExecutionError(RuntimeError):
    """Raised when symbolic expression execution fails."""


@dataclass(frozen=True)
class SymbolicChannelSpec:
    name: str
    role: str
    expression: str
    output_range_hint: Optional[Tuple[float, float]]
    rationale: str
    feature_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "expression": self.expression,
            "rationale": self.rationale,
        }
        if self.output_range_hint is not None:
            payload["output_range_hint"] = [
                float(self.output_range_hint[0]),
                float(self.output_range_hint[1]),
            ]
        return payload


@dataclass(frozen=True)
class SymbolicFeatureSpec:
    spec_version: str
    task: str
    feature_pool_used: Tuple[str, ...]
    expression: str
    output_name: str
    output_range_hint: Optional[Tuple[float, float]]
    decision_threshold: float
    decision_positive_if: str
    notes: str
    spec_hash: str
    spec_id: str
    channels: Tuple[SymbolicChannelSpec, ...] = ()
    aggregation_method: str = "weighted_sum"
    aggregation_weights: Tuple[float, ...] = (1.0,)
    aggregation_bias: float = 0.0
    aggregation_postprocess: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "spec_version": self.spec_version,
            "task": self.task,
            "feature_pool_used": list(self.feature_pool_used),
            "output_name": self.output_name,
            "decision": {
                "threshold": float(self.decision_threshold),
                "positive_if": self.decision_positive_if,
            },
            "notes": self.notes,
            "spec_hash": self.spec_hash,
            "spec_id": self.spec_id,
        }
        if self.output_range_hint is not None:
            payload["output_range_hint"] = [
                float(self.output_range_hint[0]),
                float(self.output_range_hint[1]),
            ]

        if self.spec_version == "v1":
            payload["expression"] = self.expression
        else:
            payload["channels"] = [ch.to_dict() for ch in self.channels]
            payload["aggregation"] = {
                "method": self.aggregation_method,
                "weights": [float(x) for x in self.aggregation_weights],
                "bias": float(self.aggregation_bias),
                "postprocess": self.aggregation_postprocess,
            }
            payload["expression"] = self.expression
        return payload


@dataclass(frozen=True)
class _FunctionSpec:
    min_args: int
    max_args: Optional[int]


def _fn_min(*args):
    out = _as_array(args[0])
    for x in args[1:]:
        out = np.minimum(out, _as_array(x))
    return out


def _fn_max(*args):
    out = _as_array(args[0])
    for x in args[1:]:
        out = np.maximum(out, _as_array(x))
    return out


def _fn_clip(x, lo, hi):
    return np.clip(_as_array(x), _as_array(lo), _as_array(hi))


def _fn_sigmoid(x):
    v = _as_array(x)
    return 1.0 / (1.0 + np.exp(-np.clip(v, -40.0, 40.0)))


def _fn_relu(x):
    v = _as_array(x)
    return np.maximum(v, 0.0)


def _fn_safe_div(a, b, eps=1e-6):
    numer = _as_array(a)
    denom = _as_array(b)
    eps_f = float(abs(float(eps)))
    safe = np.where(np.abs(denom) < eps_f, np.where(denom >= 0, eps_f, -eps_f), denom)
    return numer / safe


def _fn_avg(*args):
    arrs = [_as_array(x) for x in args]
    return np.sum(arrs, axis=0) / float(len(arrs))


def _fn_log(x):
    return np.log(np.clip(_as_array(x), 1e-12, None))


def _fn_log1p(x):
    return np.log1p(np.clip(_as_array(x), -0.999999, None))


def _fn_sqrt(x):
    return np.sqrt(np.clip(_as_array(x), 0.0, None))


def _fn_where(cond, x, y):
    return np.where(_as_bool_array(cond), _as_array(x), _as_array(y))


_ALLOWED_FUNCTIONS = {
    "abs": (_FunctionSpec(1, 1), np.abs),
    "min": (_FunctionSpec(2, None), _fn_min),
    "max": (_FunctionSpec(2, None), _fn_max),
    "clip": (_FunctionSpec(3, 3), _fn_clip),
    "sigmoid": (_FunctionSpec(1, 1), _fn_sigmoid),
    "relu": (_FunctionSpec(1, 1), _fn_relu),
    "tanh": (_FunctionSpec(1, 1), np.tanh),
    "log": (_FunctionSpec(1, 1), _fn_log),
    "log1p": (_FunctionSpec(1, 1), _fn_log1p),
    "exp": (_FunctionSpec(1, 1), np.exp),
    "sqrt": (_FunctionSpec(1, 1), _fn_sqrt),
    "square": (_FunctionSpec(1, 1), np.square),
    "safe_div": (_FunctionSpec(2, 3), _fn_safe_div),
    "avg": (_FunctionSpec(2, None), _fn_avg),
    "where": (_FunctionSpec(3, 3), _fn_where),
}

_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
}

_RESERVED_CONSTANTS = {"pi", "e"}


def allowed_symbolic_function_names() -> List[str]:
    return sorted(_ALLOWED_FUNCTIONS.keys())


def _as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _as_bool_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(np.float64) > 0.0


def _to_vector(value, *, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n,), float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise SymbolicExecutionError(
            f"Feature {name!r} length mismatch: expected {n}, got {arr.shape[0]}"
        )
    return arr


def _infer_length(feature_map: Mapping[str, np.ndarray]) -> int:
    for value in feature_map.values():
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        arr = arr.reshape(-1)
        if arr.shape[0] > 0:
            return int(arr.shape[0])
    return 0


def _dedup_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for name in names:
        token = str(name).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, *, allowed_features: Set[str]) -> None:
        self.allowed_features = set(str(x) for x in allowed_features)
        self.used_features: Set[str] = set()

    def generic_visit(self, node):
        if type(node) not in _ALLOWED_NODES:
            raise SymbolicSpecError(f"Unsupported syntax node: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SymbolicSpecError("Only direct function calls are allowed (no attribute calls).")
        fn_name = str(node.func.id)
        if fn_name not in _ALLOWED_FUNCTIONS:
            raise SymbolicSpecError(
                f"Unsupported function {fn_name!r}. Allowed: {allowed_symbolic_function_names()}"
            )
        if node.keywords:
            raise SymbolicSpecError("Keyword arguments are not allowed in expression functions.")
        spec, _ = _ALLOWED_FUNCTIONS[fn_name]
        argc = len(node.args)
        if argc < int(spec.min_args):
            raise SymbolicSpecError(
                f"Function {fn_name!r} expects at least {spec.min_args} args, got {argc}."
            )
        if spec.max_args is not None and argc > int(spec.max_args):
            raise SymbolicSpecError(
                f"Function {fn_name!r} expects at most {spec.max_args} args, got {argc}."
            )
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name):
        token = str(node.id)
        if token in _ALLOWED_FUNCTIONS:
            return
        if token in _RESERVED_CONSTANTS:
            return
        if token not in self.allowed_features:
            raise SymbolicSpecError(
                f"Expression references unknown feature {token!r}; allowed={sorted(self.allowed_features)}"
            )
        self.used_features.add(token)


class SymbolicExpressionProgram:
    def __init__(self, *, expression: str, allowed_features: Sequence[str]):
        self.expression = str(expression).strip()
        if not self.expression:
            raise SymbolicSpecError("expression cannot be empty")
        self.allowed_features = [str(x) for x in allowed_features]
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise SymbolicSpecError(f"Invalid expression syntax: {exc}") from exc
        validator = _ExpressionValidator(allowed_features=set(self.allowed_features))
        validator.visit(tree)
        self._tree = tree
        self.feature_names = tuple(sorted(validator.used_features))

    def evaluate(self, feature_map: Mapping[str, np.ndarray]) -> np.ndarray:
        n = _infer_length(feature_map)
        if n <= 0:
            raise SymbolicExecutionError("Feature map is empty; cannot infer sample length.")
        env: Dict[str, np.ndarray] = {}
        for name in self.feature_names:
            if name not in feature_map:
                raise SymbolicExecutionError(f"Missing feature in map: {name}")
            env[name] = _to_vector(feature_map[name], n=n, name=name)
        env["pi"] = np.full((n,), np.pi, dtype=np.float64)
        env["e"] = np.full((n,), np.e, dtype=np.float64)
        out = _eval_node(self._tree.body, env)
        vec = _to_vector(out, n=n, name="expression_output")
        return np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)


def _eval_node(node: ast.AST, env: Mapping[str, np.ndarray]):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return np.asarray(node.value, dtype=np.bool_)
        if isinstance(node.value, (int, float)):
            return np.asarray(float(node.value), dtype=np.float64)
        raise SymbolicExecutionError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        token = str(node.id)
        if token in env:
            return env[token]
        raise SymbolicExecutionError(f"Unknown symbol during evaluation: {token}")

    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -_as_array(val)
        if isinstance(node.op, ast.UAdd):
            return +_as_array(val)
        raise SymbolicExecutionError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        lhs = _eval_node(node.left, env)
        rhs = _eval_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return _as_array(lhs) + _as_array(rhs)
        if isinstance(node.op, ast.Sub):
            return _as_array(lhs) - _as_array(rhs)
        if isinstance(node.op, ast.Mult):
            return _as_array(lhs) * _as_array(rhs)
        if isinstance(node.op, ast.Div):
            return _fn_safe_div(lhs, rhs)
        if isinstance(node.op, ast.Pow):
            base = np.clip(_as_array(lhs), -1e6, 1e6)
            exp = np.clip(_as_array(rhs), -8.0, 8.0)
            return np.power(base, exp)
        if isinstance(node.op, ast.Mod):
            return np.mod(_as_array(lhs), _as_array(rhs))
        raise SymbolicExecutionError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise SymbolicExecutionError("Only single comparisons are supported.")
        lhs = _as_array(_eval_node(node.left, env))
        rhs = _as_array(_eval_node(node.comparators[0], env))
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return lhs > rhs
        if isinstance(op, ast.GtE):
            return lhs >= rhs
        if isinstance(op, ast.Lt):
            return lhs < rhs
        if isinstance(op, ast.LtE):
            return lhs <= rhs
        if isinstance(op, ast.Eq):
            return lhs == rhs
        if isinstance(op, ast.NotEq):
            return lhs != rhs
        raise SymbolicExecutionError(f"Unsupported comparison operator: {type(op).__name__}")

    if isinstance(node, ast.BoolOp):
        if not node.values:
            raise SymbolicExecutionError("Boolean operation requires operands.")
        if isinstance(node.op, ast.And):
            out = _as_bool_array(_eval_node(node.values[0], env))
            for token in node.values[1:]:
                out = out & _as_bool_array(_eval_node(token, env))
            return out
        if isinstance(node.op, ast.Or):
            out = _as_bool_array(_eval_node(node.values[0], env))
            for token in node.values[1:]:
                out = out | _as_bool_array(_eval_node(token, env))
            return out
        raise SymbolicExecutionError(f"Unsupported bool operator: {type(node.op).__name__}")

    if isinstance(node, ast.IfExp):
        cond = _as_bool_array(_eval_node(node.test, env))
        body = _as_array(_eval_node(node.body, env))
        orelse = _as_array(_eval_node(node.orelse, env))
        return np.where(cond, body, orelse)

    if isinstance(node, ast.Call):
        fn_name = str(node.func.id)
        fn_spec, fn_impl = _ALLOWED_FUNCTIONS[fn_name]
        args = [_eval_node(arg, env) for arg in node.args]
        argc = len(args)
        if argc < int(fn_spec.min_args):
            raise SymbolicExecutionError(
                f"Function {fn_name!r} expects >= {fn_spec.min_args} args, got {argc}."
            )
        if fn_spec.max_args is not None and argc > int(fn_spec.max_args):
            raise SymbolicExecutionError(
                f"Function {fn_name!r} expects <= {fn_spec.max_args} args, got {argc}."
            )
        return fn_impl(*args)

    raise SymbolicExecutionError(f"Unsupported node at eval: {type(node).__name__}")


def extract_expression_feature_names(expression: str, *, allowed_features: Sequence[str]) -> List[str]:
    program = SymbolicExpressionProgram(expression=expression, allowed_features=allowed_features)
    return list(program.feature_names)


def _normalize_output_range_hint(raw) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise SymbolicSpecError("output_range_hint must be [lo, hi].")
    lo = float(raw[0])
    hi = float(raw[1])
    if lo >= hi:
        raise SymbolicSpecError("output_range_hint must satisfy lo < hi.")
    return (lo, hi)


def _normalize_decision(raw: Any) -> Tuple[float, str]:
    decision = {} if raw is None else raw
    if not isinstance(decision, Mapping):
        raise SymbolicSpecError("decision must be an object.")
    decision_threshold = float(decision.get("threshold", 0.5))
    if decision_threshold < 0.0 or decision_threshold > 1.0:
        raise SymbolicSpecError("decision.threshold must be in [0, 1].")
    decision_positive_if = str(decision.get("positive_if", ">=")).strip() or ">="
    if decision_positive_if not in _DECISION_OPS:
        raise SymbolicSpecError(f"decision.positive_if must be one of {sorted(_DECISION_OPS)}")
    return float(decision_threshold), decision_positive_if


def _normalize_postprocess(raw: Any) -> str:
    token = str(raw if raw is not None else "none").strip().lower() or "none"
    if token in {"none", "identity", "linear", "pass"}:
        return "none"
    if token in {"sigmoid", "logistic"}:
        return "sigmoid"
    if token in {"clip01", "clip", "clip_01", "clip0to1", "clip_0_1", "clamp01", "clamp"}:
        return "clip01"
    # Keep generation robust to occasional LLM wording drift.
    return "none"


def _validate_feature_pool_constraints(
    *,
    feature_pool_used: Sequence[str],
    used_features: Sequence[str],
    allowed: Sequence[str],
) -> None:
    feature_pool_set = set(feature_pool_used)
    unknown_used = [name for name in used_features if name not in feature_pool_set]
    if unknown_used:
        raise SymbolicSpecError(
            "expression uses features not declared in feature_pool_used: " + ", ".join(sorted(unknown_used))
        )

    allowed_set = set(allowed)
    unknown_selected = [name for name in feature_pool_used if name not in allowed_set]
    if unknown_selected:
        raise SymbolicSpecError(
            "feature_pool_used contains unknown features: " + ", ".join(sorted(unknown_selected))
        )


def _validate_v1(
    doc: Mapping[str, Any],
    *,
    task: str,
    allowed: Sequence[str],
    feature_pool_used: List[str],
    output_name: str,
    output_range_hint: Optional[Tuple[float, float]],
    decision_threshold: float,
    decision_positive_if: str,
    notes: str,
) -> SymbolicFeatureSpec:
    expression = str(doc.get("expression", "")).strip()
    if not expression:
        raise SymbolicSpecError("expression is required.")

    program = SymbolicExpressionProgram(expression=expression, allowed_features=allowed)
    if len(program.feature_names) < 2:
        raise SymbolicSpecError("expression must reference at least 2 features.")

    if not feature_pool_used:
        feature_pool_used = list(program.feature_names)

    _validate_feature_pool_constraints(
        feature_pool_used=feature_pool_used,
        used_features=list(program.feature_names),
        allowed=allowed,
    )

    canonical = {
        "spec_version": "v1",
        "task": task,
        "feature_pool_used": sorted(feature_pool_used),
        "expression": expression,
        "output_name": output_name,
        "output_range_hint": list(output_range_hint) if output_range_hint is not None else None,
        "decision": {
            "threshold": float(decision_threshold),
            "positive_if": decision_positive_if,
        },
        "notes": notes,
    }
    canon_str = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(canon_str.encode("utf-8")).hexdigest()
    spec_id = f"{task}:{spec_hash[:12]}"

    return SymbolicFeatureSpec(
        spec_version="v1",
        task=task,
        feature_pool_used=tuple(sorted(feature_pool_used)),
        expression=expression,
        output_name=output_name,
        output_range_hint=output_range_hint,
        decision_threshold=float(decision_threshold),
        decision_positive_if=decision_positive_if,
        notes=notes,
        spec_hash=spec_hash,
        spec_id=spec_id,
    )


def _validate_v2(
    doc: Mapping[str, Any],
    *,
    task: str,
    allowed: Sequence[str],
    feature_pool_used: List[str],
    output_name: str,
    output_range_hint: Optional[Tuple[float, float]],
    decision_threshold: float,
    decision_positive_if: str,
    notes: str,
) -> SymbolicFeatureSpec:
    channels_raw = doc.get("channels", None)
    if not isinstance(channels_raw, list):
        raise SymbolicSpecError("v2 spec requires channels as a list.")
    if len(channels_raw) < _MIN_V2_CHANNELS or len(channels_raw) > _MAX_V2_CHANNELS:
        raise SymbolicSpecError(
            f"v2 channels count must be in [{_MIN_V2_CHANNELS}, {_MAX_V2_CHANNELS}], got {len(channels_raw)}"
        )

    channels: List[SymbolicChannelSpec] = []
    channel_name_seen: Set[str] = set()
    role_seen: Set[str] = set()
    expression_seen: Set[str] = set()
    feature_signature_seen: Set[Tuple[str, ...]] = set()
    used_union: Set[str] = set()

    for idx, raw_channel in enumerate(channels_raw):
        if not isinstance(raw_channel, Mapping):
            raise SymbolicSpecError(f"channels[{idx}] must be an object.")

        channel_name = str(raw_channel.get("name", "")).strip() or f"ch_{idx + 1}"
        if channel_name in channel_name_seen:
            raise SymbolicSpecError(f"Duplicate channel name: {channel_name!r}")
        channel_name_seen.add(channel_name)

        role = str(raw_channel.get("role", "")).strip()
        if not role:
            raise SymbolicSpecError(f"channels[{idx}].role is required.")
        role_key = role.lower()
        if role_key in role_seen:
            raise SymbolicSpecError(f"Duplicate channel role: {role!r}")
        role_seen.add(role_key)

        expression = str(raw_channel.get("expression", "")).strip()
        if not expression:
            raise SymbolicSpecError(f"channels[{idx}].expression is required.")

        program = SymbolicExpressionProgram(expression=expression, allowed_features=allowed)
        if len(program.feature_names) < 1:
            raise SymbolicSpecError(f"channels[{idx}] expression must reference at least 1 feature.")

        expr_norm = "".join(expression.split())
        if expr_norm in expression_seen:
            raise SymbolicSpecError(
                f"channels[{idx}] expression duplicates another channel (normalized match)."
            )
        expression_seen.add(expr_norm)

        feature_signature = tuple(sorted(program.feature_names))
        if feature_signature in feature_signature_seen:
            if not _ALLOW_DUP_SIGNATURE:
                raise SymbolicSpecError(
                    f"channels[{idx}] uses duplicate feature set signature {feature_signature}."
                )
        feature_signature_seen.add(feature_signature)

        used_union.update(program.feature_names)

        channel_range_hint = _normalize_output_range_hint(raw_channel.get("output_range_hint", None))
        rationale = str(raw_channel.get("rationale", "")).strip()
        channels.append(
            SymbolicChannelSpec(
                name=channel_name,
                role=role,
                expression=expression,
                output_range_hint=channel_range_hint,
                rationale=rationale,
                feature_names=tuple(sorted(program.feature_names)),
            )
        )

    if not feature_pool_used:
        feature_pool_used = sorted(used_union)

    _validate_feature_pool_constraints(
        feature_pool_used=feature_pool_used,
        used_features=sorted(used_union),
        allowed=allowed,
    )

    aggregation = doc.get("aggregation", {})
    if aggregation is None:
        aggregation = {}
    if not isinstance(aggregation, Mapping):
        raise SymbolicSpecError("aggregation must be an object for v2 specs.")

    method = str(aggregation.get("method", "weighted_sum")).strip().lower() or "weighted_sum"
    if method not in _AGGREGATION_METHODS:
        raise SymbolicSpecError(f"aggregation.method must be one of {sorted(_AGGREGATION_METHODS)}")

    weights_raw = aggregation.get("weights", None)
    if weights_raw is None:
        weights = tuple(float(1.0 / len(channels)) for _ in range(len(channels)))
    else:
        if not isinstance(weights_raw, list):
            raise SymbolicSpecError("aggregation.weights must be a list.")
        if len(weights_raw) == len(channels):
            weights = tuple(float(x) for x in weights_raw)
        elif len(weights_raw) == 1:
            weights = tuple(float(weights_raw[0]) for _ in range(len(channels)))
        else:
            # Keep generation robust: when LLM returns a short template list
            # (commonly length 4), fall back to uniform weights.
            weights = tuple(float(1.0 / len(channels)) for _ in range(len(channels)))
    for idx, w in enumerate(weights):
        if not np.isfinite(float(w)):
            raise SymbolicSpecError(f"aggregation.weights[{idx}] must be finite.")

    bias = float(aggregation.get("bias", 0.0))
    if not np.isfinite(float(bias)):
        raise SymbolicSpecError("aggregation.bias must be finite.")

    postprocess = _normalize_postprocess(aggregation.get("postprocess", "none"))

    expression = "weighted_sum(" + ",".join(ch.name for ch in channels) + ")"

    canonical_channels = []
    for channel in channels:
        canonical_channels.append(
            {
                "name": channel.name,
                "role": channel.role,
                "expression": channel.expression,
                "output_range_hint": list(channel.output_range_hint) if channel.output_range_hint is not None else None,
                "rationale": channel.rationale,
            }
        )

    canonical = {
        "spec_version": "v2",
        "task": task,
        "feature_pool_used": sorted(feature_pool_used),
        "channels": canonical_channels,
        "aggregation": {
            "method": method,
            "weights": [float(x) for x in weights],
            "bias": float(bias),
            "postprocess": postprocess,
        },
        "output_name": output_name,
        "output_range_hint": list(output_range_hint) if output_range_hint is not None else None,
        "decision": {
            "threshold": float(decision_threshold),
            "positive_if": decision_positive_if,
        },
        "notes": notes,
    }
    canon_str = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(canon_str.encode("utf-8")).hexdigest()
    spec_id = f"{task}:{spec_hash[:12]}"

    return SymbolicFeatureSpec(
        spec_version="v2",
        task=task,
        feature_pool_used=tuple(sorted(feature_pool_used)),
        expression=expression,
        output_name=output_name,
        output_range_hint=output_range_hint,
        decision_threshold=float(decision_threshold),
        decision_positive_if=decision_positive_if,
        notes=notes,
        spec_hash=spec_hash,
        spec_id=spec_id,
        channels=tuple(channels),
        aggregation_method=method,
        aggregation_weights=tuple(float(x) for x in weights),
        aggregation_bias=float(bias),
        aggregation_postprocess=postprocess,
    )


def validate_symbolic_feature_spec(
    doc: Mapping[str, Any],
    *,
    expected_task: Optional[str] = None,
    allowed_features: Optional[Sequence[str]] = None,
) -> SymbolicFeatureSpec:
    if not isinstance(doc, Mapping):
        raise SymbolicSpecError("Symbolic spec must be a JSON object.")

    task = str(doc.get("task", "")).strip()
    if not task:
        raise SymbolicSpecError("task is required in symbolic spec.")
    if expected_task is not None and task != str(expected_task):
        raise SymbolicSpecError(f"task mismatch: expected {expected_task!r}, got {task!r}")

    spec_version = str(doc.get("spec_version", SPEC_VERSION)).strip() or SPEC_VERSION
    if spec_version not in SUPPORTED_SPEC_VERSIONS:
        raise SymbolicSpecError(
            f"Unsupported spec_version={spec_version!r}, expected one of {sorted(SUPPORTED_SPEC_VERSIONS)!r}"
        )

    feature_pool_used_raw = doc.get("feature_pool_used", [])
    if feature_pool_used_raw is None:
        feature_pool_used_raw = []
    if not isinstance(feature_pool_used_raw, list):
        raise SymbolicSpecError("feature_pool_used must be a list.")
    feature_pool_used = _dedup_names([str(x) for x in feature_pool_used_raw])

    if allowed_features is not None:
        allowed = _dedup_names([str(x) for x in allowed_features])
    else:
        allowed = list(feature_pool_used)

    if not allowed:
        raise SymbolicSpecError("allowed feature pool is empty.")

    default_output_name = "sym_feature_v2" if spec_version == "v2" else "sym_feature"
    output_name = str(doc.get("output_name", default_output_name)).strip() or default_output_name
    output_range_hint = _normalize_output_range_hint(doc.get("output_range_hint", None))
    decision_threshold, decision_positive_if = _normalize_decision(doc.get("decision", {}))
    notes = str(doc.get("notes", "")).strip()

    if spec_version == "v1":
        return _validate_v1(
            doc,
            task=task,
            allowed=allowed,
            feature_pool_used=feature_pool_used,
            output_name=output_name,
            output_range_hint=output_range_hint,
            decision_threshold=decision_threshold,
            decision_positive_if=decision_positive_if,
            notes=notes,
        )

    return _validate_v2(
        doc,
        task=task,
        allowed=allowed,
        feature_pool_used=feature_pool_used,
        output_name=output_name,
        output_range_hint=output_range_hint,
        decision_threshold=decision_threshold,
        decision_positive_if=decision_positive_if,
        notes=notes,
    )


def load_symbolic_feature_spec(
    path: str | Path,
    *,
    expected_task: Optional[str] = None,
    allowed_features: Optional[Sequence[str]] = None,
) -> SymbolicFeatureSpec:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return validate_symbolic_feature_spec(
        raw,
        expected_task=expected_task,
        allowed_features=allowed_features,
    )


def save_symbolic_feature_spec(spec: SymbolicFeatureSpec, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, ensure_ascii=False, indent=2)


class SymbolicFeatureExecutor:
    def __init__(
        self,
        *,
        spec: SymbolicFeatureSpec,
        fallback_value: float = 0.0,
        strict: bool = False,
    ) -> None:
        self.spec = spec
        self.fallback_value = float(fallback_value)
        self.strict = bool(strict)

        if spec.spec_version == "v1":
            program = SymbolicExpressionProgram(
                expression=spec.expression,
                allowed_features=spec.feature_pool_used,
            )
            self.channel_programs: Tuple[SymbolicExpressionProgram, ...] = (program,)
        elif spec.spec_version == "v2":
            if not spec.channels:
                raise SymbolicSpecError("v2 spec has no channels.")
            self.channel_programs = tuple(
                SymbolicExpressionProgram(
                    expression=ch.expression,
                    allowed_features=spec.feature_pool_used,
                )
                for ch in spec.channels
            )
        else:
            raise SymbolicSpecError(f"Unsupported spec_version at executor init: {spec.spec_version!r}")

    def _run_one_program(self, *, n: int, program: SymbolicExpressionProgram, feature_map: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            out = program.evaluate(feature_map)
        except Exception:
            if self.strict:
                raise
            out = np.full((n,), self.fallback_value, dtype=np.float64)
        return np.asarray(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), dtype=np.float64)

    def run_channels(
        self,
        feature_map: Mapping[str, np.ndarray],
        *,
        apply_range_hint: bool = True,
    ) -> np.ndarray:
        n = _infer_length(feature_map)
        if n <= 0:
            raise SymbolicExecutionError("Empty feature_map; cannot execute symbolic expression.")

        if self.spec.spec_version == "v1":
            out = self._run_one_program(n=n, program=self.channel_programs[0], feature_map=feature_map)
            if apply_range_hint and self.spec.output_range_hint is not None:
                lo, hi = self.spec.output_range_hint
                out = np.clip(out, float(lo), float(hi))
            return out.reshape(-1, 1).astype(np.float32, copy=False)

        cols: List[np.ndarray] = []
        for idx, program in enumerate(self.channel_programs):
            vec = self._run_one_program(n=n, program=program, feature_map=feature_map)
            if apply_range_hint:
                channel = self.spec.channels[idx]
                if channel.output_range_hint is not None:
                    lo, hi = channel.output_range_hint
                    vec = np.clip(vec, float(lo), float(hi))
            cols.append(vec)
        stacked = np.stack(cols, axis=1)
        return stacked.astype(np.float32, copy=False)

    def run_score(
        self,
        feature_map: Mapping[str, np.ndarray],
        *,
        apply_range_hint: bool = True,
    ) -> np.ndarray:
        channels = np.asarray(self.run_channels(feature_map, apply_range_hint=apply_range_hint), dtype=np.float64)
        if channels.ndim != 2 or channels.shape[0] <= 0:
            raise SymbolicExecutionError(f"Invalid channel output shape: {channels.shape}")

        if self.spec.spec_version == "v1":
            out = channels[:, 0]
        else:
            if self.spec.aggregation_method != "weighted_sum":
                raise SymbolicExecutionError(
                    f"Unsupported aggregation_method={self.spec.aggregation_method!r}; expected 'weighted_sum'"
                )
            weights = np.asarray(self.spec.aggregation_weights, dtype=np.float64).reshape(-1)
            if weights.shape[0] != channels.shape[1]:
                raise SymbolicExecutionError(
                    "aggregation_weights length mismatch: "
                    f"weights={weights.shape[0]} channels={channels.shape[1]}"
                )
            out = channels @ weights + float(self.spec.aggregation_bias)
            post = str(self.spec.aggregation_postprocess).strip().lower() or "none"
            if post == "sigmoid":
                out = _fn_sigmoid(out)
            elif post == "clip01":
                out = np.clip(out, 0.0, 1.0)
            elif post == "none":
                pass
            else:
                raise SymbolicExecutionError(
                    f"Unsupported aggregation_postprocess={self.spec.aggregation_postprocess!r}"
                )

        if apply_range_hint and self.spec.output_range_hint is not None:
            lo, hi = self.spec.output_range_hint
            out = np.clip(out, float(lo), float(hi))
        return np.asarray(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), dtype=np.float32)

    def run(
        self,
        feature_map: Mapping[str, np.ndarray],
        *,
        apply_range_hint: bool = True,
    ) -> np.ndarray:
        return self.run_score(feature_map, apply_range_hint=apply_range_hint)


def build_feature_map_from_matrix(
    *,
    pair_features: np.ndarray,
    pair_feature_order: Sequence[str],
    extras: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    feats = np.asarray(pair_features, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"pair_features must be 2D, got shape={feats.shape}")
    names = [str(x) for x in pair_feature_order]
    if feats.shape[1] < len(names):
        raise ValueError(
            f"pair_features dim mismatch: matrix has {feats.shape[1]} cols but order has {len(names)}"
        )
    out: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        out[name] = np.asarray(feats[:, idx], dtype=np.float64)

    if extras:
        for key, value in extras.items():
            out[str(key)] = np.asarray(value, dtype=np.float64).reshape(-1)

    if "src_degree" in out and "dst_degree" in out and "degree_ratio" not in out:
        src = out["src_degree"]
        dst = out["dst_degree"]
        denom = np.maximum(src, dst)
        numer = np.minimum(src, dst)
        out["degree_ratio"] = np.where(denom > 0, numer / denom, 0.0)

    if "gnn_score" in out and "uncertainty" not in out:
        s = np.asarray(out["gnn_score"], dtype=np.float64)
        out["uncertainty"] = 1.0 - np.abs(s - 0.5) * 2.0

    return out


def apply_decision_rule(scores: np.ndarray, *, threshold: float, positive_if: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    thr = float(threshold)
    op = str(positive_if).strip()
    if op == ">=":
        pred = s >= thr
    elif op == ">":
        pred = s > thr
    elif op == "<=":
        pred = s <= thr
    elif op == "<":
        pred = s < thr
    else:
        raise ValueError(f"Unsupported positive_if operator: {positive_if!r}")
    return pred.astype(np.int64)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float, positive_if: str = ">=") -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = apply_decision_rule(scores, threshold=threshold, positive_if=positive_if)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall / (precision + recall))
    acc = float(tp + tn) / float(max(1, y.size))
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "threshold": float(threshold),
    }


def best_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    positive_if: str = ">=",
    max_candidates: int = 2048,
) -> Tuple[float, float]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    unique_scores = np.unique(s)
    if unique_scores.size == 0:
        return 0.5, 0.0
    if unique_scores.size > int(max_candidates):
        q = np.linspace(0.0, 1.0, int(max_candidates), dtype=np.float64)
        cands = np.quantile(s, q)
    else:
        cands = unique_scores

    best_thr = float(cands[0])
    best_f1 = -1.0
    for thr in cands:
        f1 = binary_metrics(y, s, threshold=float(thr), positive_if=positive_if)["f1"]
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and float(thr) < best_thr):
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, max(0.0, float(best_f1))


def strip_code_fences(text: str) -> str:
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = strip_code_fences(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise SymbolicSpecError("Extracted JSON is not an object.")
        return parsed
    raise SymbolicSpecError("Cannot extract JSON object from model output.")
