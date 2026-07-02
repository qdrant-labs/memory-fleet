"""Two-tier matching decision (PLAN.md §3.4).

Pure: candidates in, decision out. The negative-exemplar veto and the
human-provenance bonus are the WHOLE rerank — no color signatures, no class
gates (they patched detector-label problems that no longer exist).
"""

from dataclasses import dataclass

import numpy as np

from .store import Candidate

# Measured operating points (§9.2), live-tunable in the UI.
S_SAME = 0.80
S_SUGGEST = 0.55
S_IGNORE = 0.90
HUMAN_BONUS = 0.02  # a human-vouched view is worth a nudge at the margin


@dataclass(slots=True)
class Thresholds:
    s_same: float = S_SAME
    s_suggest: float = S_SUGGEST
    s_ignore: float = S_IGNORE


@dataclass(slots=True)
class Decision:
    outcome: str  # "recognized" | "suggest" | "unknown" | "ignored"
    candidate: Candidate | None = None
    score: float = 0.0  # adjusted score backing the outcome


def _neg_veto(vec: np.ndarray, cand: Candidate, session_negs: dict[str, list]) -> bool:
    """True if a 'not me' vector explains this query better than the match does."""
    negs = list(cand.payload.get("neg") or []) + list(session_negs.get(cand.id) or [])
    if not negs:
        return False
    neg_sim = max(float(vec @ np.asarray(n, dtype=np.float32)) for n in negs)
    return neg_sim >= cand.score


def decide(
    vec: np.ndarray,
    candidates: list[Candidate],
    thresholds: Thresholds,
    vetoed_ids: set[str] | None = None,
    session_negs: dict[str, list] | None = None,
) -> Decision:
    vetoed_ids = vetoed_ids or set()
    session_negs = session_negs or {}

    best: tuple[float, Candidate] | None = None
    for cand in candidates:
        if cand.kind == "ignored":
            # blocklist: strict threshold — a false suppression is the costly error
            if cand.score >= thresholds.s_ignore:
                return Decision("ignored", cand, cand.score)
            continue
        if cand.id in vetoed_ids or _neg_veto(vec, cand, session_negs):
            continue
        adjusted = cand.score
        if any(v.get("human") for v in cand.payload.get("views") or []):
            adjusted += HUMAN_BONUS
        if best is None or adjusted > best[0]:
            best = (adjusted, cand)

    if best is None:
        return Decision("unknown")
    adjusted, cand = best
    if adjusted >= thresholds.s_same:
        return Decision("recognized", cand, adjusted)
    if adjusted >= thresholds.s_suggest:
        return Decision("suggest", cand, adjusted)
    return Decision("unknown", cand, adjusted)
