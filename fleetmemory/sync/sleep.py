"""The fleet sleeps: consolidate duplicate instances, decay stale memories.

Consolidation folds same-label fleet points that plausibly ARE the same
physical thing (cross max-sim >= S_SUGGEST, the gate teach and push folds
use) into the oldest sibling. Decay scores every point with a Qdrant
exp_decay formula on t_seen and moves everything below the floor into
<collection>-archive — the hot fleet stays small, so every unit's mirror
(a snapshot of it) stays small too. Archived memories are cold, not gone.

Run while units are quiet — it rewrites fleet points and a concurrent push
into a point being folded can lose that push:

  uv run python -m fleetmemory.sync.sleep [--half-life-days 30] [--floor 0.05] [--dry-run]
"""

import argparse
import logging
import time

import numpy as np
from qdrant_client import models

from fleetmemory.memory.core import NEG_CAP, fold_rows
from fleetmemory.memory.matcher import S_SUGGEST
from fleetmemory.sync.client import FleetClient

logger = logging.getLogger(__name__)

HALF_LIFE_DAYS = 30.0
FLOOR = 0.05  # exp_decay score below this = archived (~130 days unseen at 30d half-life)


def _scroll_all(client: FleetClient, with_vectors: bool = True) -> list:
    recs, offset = [], None
    while True:
        page, offset = client.client.scroll(
            client.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        recs += page
        if offset is None:
            return recs


def _maxsim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(a @ b.T))


def _born(rec) -> float:
    pl = rec.payload or {}
    return pl.get("t_created") or pl.get("t_sync") or 0.0


def consolidate(client: FleetClient, dry_run: bool = False) -> int:
    """Merge same-label duplicates into the oldest sibling. Returns points folded."""
    groups: dict[str, list] = {}
    for rec in _scroll_all(client):
        if (rec.vector or {}).get("exemplars"):
            groups.setdefault((rec.payload or {}).get("label_key", ""), []).append(rec)

    folded = 0
    for label_key, recs in groups.items():
        if not label_key or len(recs) < 2:
            continue
        recs.sort(key=_born)  # oldest first: it survives, keeping ids stable for mirrors
        mats = [np.asarray(r.vector["exemplars"], dtype=np.float32) for r in recs]
        gone: set[int] = set()
        for i, keep in enumerate(recs):
            if i in gone:
                continue
            ate = False
            for j in range(i + 1, len(recs)):
                if j in gone or _maxsim(mats[i], mats[j]) < S_SUGGEST:
                    continue
                kp, fp = keep.payload or {}, recs[j].payload or {}
                rows, views = fold_rows(
                    list(mats[i]),
                    list(kp.get("views") or []),
                    list(mats[j]),
                    list(fp.get("views") or []),
                )
                mats[i] = np.asarray(rows, dtype=np.float32)
                keep.payload = {
                    **kp,
                    "views": views,
                    "neg": (list(kp.get("neg") or []) + list(fp.get("neg") or []))[-NEG_CAP:],
                    "t_seen": max(kp.get("t_seen") or 0, fp.get("t_seen") or 0) or None,
                    "thumb": kp.get("thumb") or fp.get("thumb", ""),
                }
                gone.add(j)
                ate = True
                logger.info("fold %r: %s <- %s", label_key, keep.id, recs[j].id)
            if ate and not dry_run:
                vector = dict(keep.vector)
                vector["exemplars"] = [list(map(float, r)) for r in mats[i]]
                client.client.upsert(
                    client.collection,
                    points=[models.PointStruct(id=keep.id, vector=vector, payload=keep.payload)],
                )
        if gone and not dry_run:
            client.client.delete(
                client.collection,
                points_selector=models.PointIdsList(points=[recs[j].id for j in gone]),
            )
        folded += len(gone)
    return folded


def decay(
    client: FleetClient,
    half_life_days: float = HALF_LIFE_DAYS,
    floor: float = FLOOR,
    now: float | None = None,
    dry_run: bool = False,
) -> int:
    """Archive points whose exp_decay(t_seen) recency score sank below the
    floor. The scoring runs on Qdrant via a formula query; points that never
    got a t_seen stamp fall back to t_sync via a backfill first."""
    now = now or time.time()
    total = client.client.count(client.collection).count
    if not total:
        return 0

    # backfill: pre-heartbeat points decay from their push time, not from zero
    for rec in _scroll_all(client, with_vectors=False):
        pl = rec.payload or {}
        if not pl.get("t_seen") and not dry_run:
            client.client.set_payload(
                client.collection,
                payload={"t_seen": pl.get("t_sync") or pl.get("t_created") or now},
                points=[rec.id],
            )

    # rank every point by recency on the server: a query-less prefetch scrolls
    # the collection, the exp_decay formula rescores it (score 1 = just seen,
    # 0.5 at one half-life, -> 0 as memories go stale)
    scored = client.client.query_points(
        client.collection,
        prefetch=models.Prefetch(limit=total),
        query=models.FormulaQuery(
            formula=models.ExpDecayExpression(
                exp_decay=models.DecayParamsExpression(
                    x="t_seen", target=now, scale=half_life_days * 86400.0
                )
            ),
            # only reachable on --dry-run (real runs backfill first): points
            # never stamped read as fresh, so a preview never over-reports
            defaults={"t_seen": now},
        ),
        limit=total,
        with_payload=True,
        with_vectors=True,
    ).points
    stale = [p for p in scored if p.score < floor]
    if not stale or dry_run:
        return len(stale)

    archive = FleetClient(
        client.url, client.api_key, collection=f"{client.collection}-archive", dim=client.dim
    )
    archive.ensure_collection()
    archive.client.upsert(
        archive.collection,
        points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in stale],
    )
    client.client.delete(
        client.collection,
        points_selector=models.PointIdsList(points=[p.id for p in stale]),
    )
    return len(stale)


def main():
    from fleetmemory.config import load_settings

    ap = argparse.ArgumentParser(description="consolidate + decay the fleet collection")
    ap.add_argument("--half-life-days", type=float, default=HALF_LIFE_DAYS)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    if not settings.fleet_enabled:
        raise SystemExit("no QDRANT_URL in .env — nothing to sleep on")
    client = FleetClient(settings.qdrant_url, settings.qdrant_api_key)
    client.ensure_collection()

    folded = consolidate(client, dry_run=args.dry_run)
    stale = decay(
        client, half_life_days=args.half_life_days, floor=args.floor, dry_run=args.dry_run
    )
    left = client.client.count(client.collection).count
    verb = "would fold" if args.dry_run else "folded"
    print(f"{verb} {folded} duplicate(s), archived {stale} stale point(s); fleet holds {left}")


if __name__ == "__main__":
    main()
