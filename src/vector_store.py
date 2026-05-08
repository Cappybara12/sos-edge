from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)

from src.config import (
    COLLECTION_NAME,
    SHARD_DIRECTORY,
    VECTOR_DIMENSION,
    VECTOR_NAME,
    SIMILARITY_THRESHOLD,
)


class VectorStore:

    def __init__(self, shard_dir: str = SHARD_DIRECTORY) -> None:
        self._shard_dir = Path(shard_dir)
        self._shard: Optional[EdgeShard] = None

    def open(self) -> None:
        self._shard_dir.mkdir(parents=True, exist_ok=True)

        existing_data = any(self._shard_dir.iterdir()) if self._shard_dir.exists() else False

        if existing_data:
            print(f"[vector_store] Loading existing shard from {self._shard_dir} ...")
            self._shard = EdgeShard.load(str(self._shard_dir))
            print("[vector_store] Shard loaded")
        else:
            print(f"[vector_store] Creating new shard at {self._shard_dir} ...")
            config = EdgeConfig(
                vectors={
                    VECTOR_NAME: EdgeVectorParams(
                        size=VECTOR_DIMENSION,
                        distance=Distance.Cosine,
                    )
                }
            )
            self._shard = EdgeShard.create(str(self._shard_dir), config)

            self._shard.update(
                UpdateOperation.create_field_index("alert_class", PayloadSchemaType.Keyword)
            )
            self._shard.update(
                UpdateOperation.create_field_index("sound_type", PayloadSchemaType.Keyword)
            )
            self._shard.update(
                UpdateOperation.create_field_index("severity", PayloadSchemaType.Keyword)
            )
            print("[vector_store] New shard created with payload indexes")

    def close(self) -> None:
        if self._shard is not None:
            self._shard.close()
            self._shard = None
            print("[vector_store] Shard closed and flushed to disk")

    def optimize(self) -> None:
        if self._shard is not None:
            print("[vector_store] Running optimizer...")
            self._shard.optimize()
            print("[vector_store] Optimization complete")

    def upsert(self, embedding: np.ndarray, payload: dict) -> str:
        assert self._shard is not None, "Call open() first"
        point_id = uuid.uuid4()
        point = Point(
            id=str(point_id),
            vector={VECTOR_NAME: embedding.tolist()},
            payload=payload,
        )
        self._shard.update(UpdateOperation.upsert_points([point]))
        return str(point_id)

    def upsert_batch(self, embeddings: list[np.ndarray], payloads: list[dict]) -> list[str]:
        assert self._shard is not None, "Call open() first"
        assert len(embeddings) == len(payloads)

        points = []
        ids = []
        for emb, payload in zip(embeddings, payloads):
            pid = uuid.uuid4()
            ids.append(str(pid))
            points.append(
                Point(
                    id=str(pid),
                    vector={VECTOR_NAME: emb.tolist()},
                    payload=payload,
                )
            )

        self._shard.update(UpdateOperation.upsert_points(points))
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        only_alerts: bool = True,
    ) -> list[dict]:
        assert self._shard is not None, "Call open() first"

        search_filter = None
        if only_alerts:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="alert_class",
                        match=MatchValue(value="alert"),
                    )
                ]
            )

        results = self._shard.query(
            QueryRequest(
                query=Query.Nearest(query_embedding.tolist(), using=VECTOR_NAME),
                filter=search_filter,
                limit=top_k,
                with_vector=False,
                with_payload=True,
            )
        )

        return [
            {
                "score": r.score,
                "payload": r.payload if hasattr(r, "payload") else {},
            }
            for r in results
        ]

    def count(self) -> int:
        assert self._shard is not None, "Call open() first"
        return self._shard.count(CountRequest(exact=False))

    def info(self) -> dict:
        assert self._shard is not None, "Call open() first"
        return self._shard.info()


if __name__ == "__main__":
    import numpy as np

    print("Running vector_store smoke test...")
    store = VectorStore(shard_dir="./test-shard")
    store.open()

    fake_emb = np.random.rand(VECTOR_DIMENSION).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)
    store.upsert(fake_emb, {"alert_class": "alert", "sound_type": "scream", "severity": "high"})

    noise_emb = np.random.rand(VECTOR_DIMENSION).astype(np.float32)
    noise_emb /= np.linalg.norm(noise_emb)
    store.upsert(noise_emb, {"alert_class": "negative", "sound_type": "music", "severity": "none"})

    print(f"Total points: {store.count()}")

    results = store.search(fake_emb, top_k=3)
    print(f"Top results: {results}")

    store.close()

    import shutil
    shutil.rmtree("./test-shard", ignore_errors=True)
    print("Smoke test passed ✓")
