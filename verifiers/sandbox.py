"""Restricted executor for VLM-written perception check snippets (Mechanism B).

A snippet may call ONLY the bound primitives (clip, clip_at, duration) plus a
small set of safe builtins, and must assign a float `score` (signed evidence for
the candidate answer: >0 supports, ~0 inconclusive). Anything else — syntax
error, exception, timeout, missing / NaN score — is treated as ABSTAIN. No file,
network, or import access is reachable from inside the snippet (stripped
`__builtins__`, no `open`/`os`/`__import__`).

`signal.alarm` is used for the wall-clock timeout, so run_check MUST be called
from the main thread (verify_clip_checks.py executes serially in phase 2).
"""
from __future__ import annotations

import ast
import math
import signal
from typing import Any, Dict, Optional, Tuple

_B = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_SAFE_BUILTINS = {
    k: _B[k]
    for k in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
              "isinstance", "len", "list", "map", "max", "min", "print", "range",
              "round", "sorted", "str", "sum", "tuple", "zip")
    if k in _B
}


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


def run_check(snippet: str, primitives: Dict[str, Any],
              timeout: int = 5) -> Tuple[Optional[float], str]:
    """Execute a check snippet. Returns (score, status).

    score is None on abstain; status in
    {ok, empty, parse_error, timeout, exec_error, no_score}.
    """
    if not snippet or not snippet.strip():
        return None, "empty"
    try:
        ast.parse(snippet)
    except SyntaxError:
        return None, "parse_error"
    ns: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    ns.update(primitives)
    old = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(int(timeout))
        exec(compile(snippet, "<check>", "exec"), ns)
    except _Timeout:
        return None, "timeout"
    except Exception:
        return None, "exec_error"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    v = ns.get("score", None)
    if isinstance(v, bool):  # tolerate a bool result -> small signed margin proxy
        return (0.05 if v else -0.05), "ok"
    if v is None:
        return None, "no_score"
    try:
        f = float(v)
    except Exception:
        return None, "no_score"
    if math.isnan(f) or math.isinf(f):
        return None, "no_score"
    return f, "ok"
