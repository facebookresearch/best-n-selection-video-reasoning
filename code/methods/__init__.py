"""methods/__init__.py — dispatch to per-method Method class."""

from __future__ import annotations

from typing import Any, Dict, Type


def get_method_class(name: str) -> Type[Any]:
    if name == "zeroshot_cot":
        from .zeroshot_cot import Method

        return Method
    if name == "jcef":
        from .jcef import Method

        return Method
    if name == "llovi":
        from .llovi import Method

        return Method
    if name == "videoagent":
        from .videoagent import Method

        return Method
    if name == "vap":
        from .vap import Method

        return Method
    if name == "tama":
        from .tama import Method

        return Method
    if name == "evidence_loop":
        from .evidence_loop import Method

        return Method
    if name == "best_of_n":
        from .best_of_n import Method

        return Method
    raise ValueError(
        f"unknown method: {name} (known: zeroshot_cot, jcef, llovi, videoagent, vap, tama, evidence_loop)"
    )
