"""On-device instance embeddings (PLAN.md §9.2: Unicom-ViT-B-32 via fastembed, ONNX CPU).

CPU on purpose — the detector owns MPS. 512-d, ~7.4 ms/crop measured.
"""

import numpy as np

MODEL_NAME = "Qdrant/Unicom-ViT-B-32"
DIM = 512


class Embedder:
    def __init__(self):
        self._model = None

    def load(self):
        if self._model is None:
            from fastembed import ImageEmbedding

            self._model = ImageEmbedding(MODEL_NAME)

    def embed(self, crops: list) -> list[np.ndarray]:
        """PIL crops -> L2-normalized float32 vectors (DIM,)."""
        self.load()
        out = []
        for v in self._model.embed(crops, batch_size=16):
            v = np.asarray(v, dtype=np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out
