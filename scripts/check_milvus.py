"""Round-trip a vector through the real Milvus using the app's own settings.

The ReID path is the one part of SightIndex that cannot be exercised without a live Milvus, so
this is the first thing to run after starting one:

    .venv/bin/python scripts/check_milvus.py
    .venv/bin/python scripts/check_milvus.py --object-type person_crop

It writes into the configured collection prefix and deletes what it wrote.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# app.db.session calls get_settings() at import time and it is lru_cached, so this default has
# to land before any app module is imported or it silently has no effect.
os.environ.setdefault("MILVUS_ENABLED", "true")

from app.config.settings import get_settings  # noqa: E402
from app.services.vector_index import MilvusVectorIndex, VectorIndexError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-type",
        default="reid_person_crop",
        help="collection to probe (default: reid_person_crop)",
    )
    args = parser.parse_args()

    settings = get_settings()
    index = MilvusVectorIndex(settings)

    print(f"host              {settings.milvus_host}:{settings.milvus_port}")
    print(f"database          {settings.milvus_db}")
    print(f"metric            {settings.milvus_metric_type}")
    print(f"object type       {args.object_type}")

    # is_available(), not is_enabled(): raw-vector callers like ReID bring their own vectors
    # and must not require a text or visual embedding provider to be configured.
    if not index.is_available():
        print("\nMILVUS_ENABLED is off, pymilvus is missing, or Milvus is in failure cooldown.")
        return 2

    probe_id: uuid.UUID | None = None
    try:
        dim = index._embedding_dim(args.object_type)
        collection = index._collection_name(args.object_type)
        print(f"collection        {collection}")
        print(f"dimension         {dim}")

        # A unit vector along one axis: cosine against itself is exactly 1.0, which makes a
        # wrong metric or a truncated vector obvious rather than approximately fine.
        probe_id = uuid.uuid4()
        vector = [0.0] * dim
        vector[0] = 1.0

        index.upsert_vector(args.object_type, probe_id, vector, "milvus smoke check", flush=True)
        print("upsert            ok")

        hits = index.search_vector(args.object_type, vector, 5) or []
        found = next((hit for hit in hits if hit.object_id == probe_id), None)
        if found is None:
            print(f"\nsearch did not return the vector just written ({len(hits)} other hits).")
            return 1
        print(f"search            ok, self-similarity {found.score:.6f}")
        if abs(found.score - 1.0) > 0.01:
            print("\nSelf-similarity should be 1.0 for COSINE; check MILVUS_METRIC_TYPE.")
            return 1
    except VectorIndexError as exc:
        print(f"\nMilvus error: {exc}")
        return 1
    finally:
        if probe_id is not None:
            try:
                index._collection(args.object_type).delete(
                    expr=f'object_id == "{probe_id}"',
                    timeout=settings.milvus_timeout_seconds,
                )
            except Exception:
                pass

    print("\nMilvus round-trip passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
