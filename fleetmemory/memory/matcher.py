"""Two-tier matching decision.

Pure: candidates in, decision out. The rerank is exactly two rules — a
negative-exemplar veto (a "not me" vector that explains the query better than
the match vetoes it) and a small human-provenance bonus at the margin.
"""

from dataclasses import dataclass

import numpy as np

from .store import Candidate

# Default operating points, live-tunable in the UI.
S_SAME = 0.80
S_SUGGEST = 0.55
S_IGNORE = 0.90
HUMAN_BONUS = 0.02  # a human-vouched view is worth a nudge at the margin
# Soft suppression: a look this close to an ignored entry is hidden too —
# UNLESS memory has a better idea (a taught object outranking it wins).
# Keeps ignored doors/hair from re-flooding the unknowns queue while staying
# visible as a faint box, one click from rescue (asymmetric suppression).
S_IGNORE_SOFT = 0.65


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
    best_ignored: Candidate | None = None
    for cand in candidates:
        if cand.kind == "ignored":
            if cand.score >= thresholds.s_ignore:
                return Decision("ignored", cand, cand.score)  # hard match
            if best_ignored is None or cand.score > best_ignored.score:
                best_ignored = cand
            continue
        if cand.id in vetoed_ids or _neg_veto(vec, cand, session_negs):
            continue
        adjusted = cand.score
        if any(v.get("human") for v in cand.payload.get("views") or []):
            adjusted += HUMAN_BONUS
        if best is None or adjusted > best[0]:
            best = (adjusted, cand)

    soft_ignore = best_ignored is not None and best_ignored.score >= S_IGNORE_SOFT
    if best is None:
        if soft_ignore:
            return Decision("ignored", best_ignored, best_ignored.score)
        return Decision("unknown")
    adjusted, cand = best
    if adjusted >= thresholds.s_same:
        return Decision("recognized", cand, adjusted)  # a taught object outranks a look-alike
    if soft_ignore and best_ignored.score > adjusted:
        return Decision("ignored", best_ignored, best_ignored.score)
    if adjusted >= thresholds.s_suggest:
        return Decision("suggest", cand, adjusted)
    return Decision("unknown", cand, adjusted)
