"""Neural label embeddings for hybrid search: miniCOIL sparse + dense text.

fastembed lives in the perception extra, so this loads lazily. Without a
LabelEmbedder, Store falls back to on-device BM25 (same sparse field, IDF
modifier).
"""

from qdrant_edge import SparseVector

SPARSE_MODEL = "Qdrant/minicoil-v1"  # needs the IDF modifier on the sparse field
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384


class LabelEmbedder:
    def __init__(self):
        self._sparse = None
        self._dense = None

    def load(self):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding, TextEmbedding

            self._sparse = SparseTextEmbedding(SPARSE_MODEL)
            self._dense = TextEmbedding(DENSE_MODEL)

    @staticmethod
    def _sv(e) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in e.indices], values=[float(v) for v in e.values]
        )

    def embed_doc(self, text: str) -> tuple[SparseVector, list[float]]:
        self.load()
        text = text.strip() or " "
        sp = next(iter(self._sparse.embed([text])))
        dv = next(iter(self._dense.embed([text])))
        return self._sv(sp), [float(x) for x in dv]

    def embed_query(self, text: str) -> tuple[SparseVector, list[float]]:
        self.load()
        text = text.strip() or " "
        sp = next(iter(self._sparse.query_embed(text)))
        dv = next(iter(self._dense.query_embed(text)))
        return self._sv(sp), [float(x) for x in dv]
