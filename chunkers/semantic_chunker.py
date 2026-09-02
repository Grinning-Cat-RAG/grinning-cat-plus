from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from cat import Embeddings

from .constants import DISPLAY_FORMULA, INLINE_FORMULA, LATEX_ENV


class BaseSemanticChunker(ABC):
    def __init__(self, max_tokens: int, cluster_threshold: float, similarity_threshold: float):
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self._embedder: Optional[Embeddings] = None

    @property
    def embedder(self):
        return self._embedder

    @embedder.setter
    def embedder(self, embedder: Embeddings):
        self._embedder = embedder

    def _token_budget(self) -> int:
        """Chunk token budget: the embedder's max_input_tokens when declared,
        else the configured max_tokens.

        This makes the chunker aware of the configured embedder's input ceiling
        so merged chunks never exceed it; the core rabbit-hole also re-checks the
        final list against embedder.max_input_tokens as an extra guard.
        """
        embedder_budget = getattr(self._embedder, "max_input_tokens", None)
        if embedder_budget and embedder_budget > 0:
            # leave small headroom for batching/tokenizer overestimation
            return max(1, int(embedder_budget * 0.9))
        return self.max_tokens

    def _count_tokens(self, text: str) -> int:
        """Count tokens for a chunk, preferring the embedder's own estimator so
        the count matches the model that will embed it."""
        if self._embedder is not None and hasattr(self._embedder, "_estimate_tokens"):
            return self._embedder._estimate_tokens(text)
        # fallback: conservative ~3 chars/token, never an undercount
        return max(1, len(text) // 3)

    def _fit_to_embed_budget(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pre-split any chunk whose text is too large for the embedder.

        The clustering step embeds every chunk; if one chunk alone exceeds the
        embedder's input window the request fails. We therefore split oversized
        chunks into budget-sized sub-chunks HERE, before any embedding, so the
        full text is always chunked before it is ever sent to the embedder.
        Sub-chunks inherit the metadata; the downstream merge step re-joins them
        (and applies ``_token_budget``) so the final output size is unaffected.
        """
        budget = self._token_budget()
        if budget <= 0:
            return chunks

        fitted: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text or self._count_tokens(text) <= budget:
                fitted.append(chunk)
                continue

            # Exact, linear split: coarse chars/estimate pass, then refine each
            # overshooting candidate by measuring word-by-word near the boundary
            # (does NOT re-encode the whole growing candidate at every word).
            words = text.split()
            # avg tokens per word for THIS text (sampled on 200 words)
            sample_tokens = max(1, self._count_tokens(" ".join(words[:200])))
            approx_tpw = sample_tokens / max(1, len(words[:200]))
            per_chunk_words = max(1, int(budget / approx_tpw))

            for start in range(0, len(words), per_chunk_words):
                part = words[start:start + per_chunk_words]
                text_part = " ".join(part)
                if self._count_tokens(text_part) <= budget:
                    fitted.append({**chunk, "text": text_part})
                    continue
                # refine: shave words until exactly within budget (linear)
                lo, hi = 0, len(part)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._count_tokens(" ".join(part[:mid])) <= budget:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo == 0:
                    # single word alone exceeds budget (pathological): emit it
                    lo = 1
                fitted.append({**chunk, "text": " ".join(part[:lo])})
                # push the remainder into the next slice, if any
                if lo < len(part):
                    words[start + lo: start + len(part)] = part[lo:]
        return fitted

    def _calculate_clusters(self, adjusted: np.ndarray, n: int) -> List:
        def find(x: int):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            parent[find(x)] = find(y)

        parent = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if adjusted[i, j] >= self.cluster_threshold:
                    union(i, j)

        clusters = [find(i) for i in range(n)]
        cluster_map = {cid: k for k, cid in enumerate(sorted(set(clusters)))}
        return [cluster_map[c] for c in clusters]

    def chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        # Always chunk the full text before any embedding: split oversized
        # chunks into embed-budget-sized sub-chunks here, so the clustering
        # embed (which runs on the whole input list) never sees text that
        # exceeds the embedder's input window.
        chunks = self._fit_to_embed_budget(chunks)

        return self._chunk(chunks)

    @abstractmethod
    def _chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class SemanticChunker(BaseSemanticChunker):
    def _chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # calculate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = np.array(self.embedder.embed_documents(texts))

        # compute similarity
        similarity_matrix = np.zeros((0, 0)) if embeddings.size == 0 else cosine_similarity(embeddings)

        # calculate clusters
        clusters = self._calculate_clusters(similarity_matrix, similarity_matrix.shape[0])

        # merge chunks
        if not chunks or not clusters:
            return []

        cluster_map = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_map[cluster_id].append(chunks[idx])

        merged_chunks = []
        for chunk_list in cluster_map.values():
            current_text = ""
            current_meta = []

            for chunk in chunk_list:
                next_text = (current_text + " " + chunk["text"]).strip()
                num_tokens = self._count_tokens(next_text)

                if current_text and num_tokens > self._token_budget():
                    merged_chunks.append({
                        "text": current_text,
                        "metadata": current_meta
                    })
                    current_text = chunk["text"]
                    current_meta = [chunk]
                else:
                    current_text = next_text
                    current_meta.append(chunk)

            if current_text:
                merged_chunks.append({
                    "text": current_text,
                    "metadata": current_meta
                })

        return merged_chunks


class MathAwareSemanticChunker(BaseSemanticChunker):
    """
    Math-aware semantic chunker that combines the semantic clustering approach
    of SemanticChunker with formula-awareness inspired by MathAwareHierarchicalChunker.

    Key behaviours:
    1. Formulas (display ``$$…$$``, LaTeX environments, inline ``$…$``) are replaced
       with unique placeholders *before* embeddings are computed, so semantic similarity
       is driven by the surrounding prose rather than raw formula symbols.
    2. Adjacent chunk pairs where at least one carries a formula receive a configurable
       similarity boost (``formula_context_boost``) so they tend to cluster together and
       the formula is never left without its explanatory context.
    3. After the union-find merge step, placeholders are restored to their original
       formula strings and each output chunk is annotated with ``has_formula`` and
       ``formula_count`` metadata fields.
    """
    def __init__(
        self,
        max_tokens: int,
        cluster_threshold: float,
        similarity_threshold: float,
        formula_context_boost: float = 0.3,
    ):
        super().__init__(max_tokens, cluster_threshold, similarity_threshold)
        self.formula_context_boost = formula_context_boost

    @staticmethod
    def _protect_formulas(text: str, chunk_index: int = 0) -> Tuple[str, Dict[str, str]]:
        """
        Replace every formula in *text* with a unique placeholder.
        Placeholders encode the chunk index so they never collide across chunks.
        Returns ``(protected_text, formula_map)`` where ``formula_map`` maps each
        placeholder back to the original formula string.
        """
        formula_map: Dict[str, str] = {}
        protected = text
        counter = 0

        # Order matters: display > LaTeX env > inline (avoids partial matches)
        patterns = [
            (DISPLAY_FORMULA, re.DOTALL),
            (LATEX_ENV, re.DOTALL),
            (INLINE_FORMULA, 0),
        ]
        for pattern, flags in patterns:
            for match in re.finditer(pattern, text, flags):
                formula = match.group(0)
                placeholder = f"__FORMULA_{chunk_index}_{counter}__"
                formula_map[placeholder] = formula
                protected = protected.replace(formula, placeholder, 1)
                counter += 1

        return protected, formula_map

    @staticmethod
    def _restore_formulas(text: str, formula_map: Dict[str, str]) -> str:
        """Substitute all placeholders back with their original formula strings."""
        restored = text
        for placeholder, formula in formula_map.items():
            restored = restored.replace(placeholder, formula)
        return restored

    def _chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _flush() -> None:
            restored = self._restore_formulas(current_text, current_fmap)
            formula_cnt = len(current_fmap)
            merged_chunks.append({
                "text": restored,
                "metadata": current_meta,
                "has_formula": formula_cnt > 0,
                "formula_count": formula_cnt,
            })

        # ---- Step 1: protect formulas per chunk with unique placeholders
        formula_maps: List[Dict[str, str]] = []
        protected_chunks: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            p_text, f_map = self._protect_formulas(chunk["text"], chunk_index=idx)
            formula_maps.append(f_map)
            protected_chunks.append({**chunk, "text": p_text})

        # ---- Step 2: compute embeddings on formula-protected text
        texts = [c["text"] for c in protected_chunks]
        embeddings = np.array(self.embedder.embed_documents(texts))

        # ---- Step 3: cosine similarity + formula-context boost
        n = len(protected_chunks)
        similarity_matrix = cosine_similarity(embeddings)

        has_formula_flags = [bool(fm) for fm in formula_maps]

        adjusted = similarity_matrix.copy()
        for i in range(n):
            for j in range(i + 1, n):
                # Boost similarity between *adjacent* chunks when either has a formula
                if (has_formula_flags[i] or has_formula_flags[j]) and abs(i - j) == 1:
                    boosted = adjusted[i, j] + self.formula_context_boost
                    adjusted[i, j] = min(boosted, 1.0)
                    adjusted[j, i] = adjusted[i, j]

        # ---- Step 4: union-find clustering
        cluster_ids = self._calculate_clusters(adjusted, n)

        # ---- Step 5: group by cluster (preserving original order)
        cluster_groups: Dict[int, List[Tuple[int, Dict[str, Any], Dict[str, str]]]] = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            cluster_groups[cid].append((idx, protected_chunks[idx], formula_maps[idx]))

        # ---- Step 6: merge within each cluster, then restore formulas
        merged_chunks: List[Dict[str, Any]] = []

        for group in cluster_groups.values():
            current_text: str = ""
            current_meta: List[Any] = []
            current_fmap: Dict[str, str] = {}

            for _idx, chunk, f_map in group:
                next_text = (current_text + " " + chunk["text"]).strip()
                num_tokens = self._count_tokens(next_text)

                if current_text and num_tokens > self._token_budget():
                    _flush()
                    current_text = chunk["text"]
                    current_meta = [chunk]
                    current_fmap = dict(f_map)
                else:
                    current_text = next_text
                    current_meta.append(chunk)
                    current_fmap.update(f_map)

            if current_text:
                _flush()

        return merged_chunks
