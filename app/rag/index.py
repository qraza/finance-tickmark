"""Document retrieval over the corpus, with citable chunks.

Chunks are paragraphs, labelled with their document and heading, so one monthly
commentary yields separate passages for each variance it explains. Commentary
chunks carry their month, so a search can be scoped to the period the question is
about (taken from the semantic query Claude chose). Ranking is BM25, pure Python.

Embeddings (hybrid search) are deliberately not here yet: they go in only if they
beat this on the golden set. See `make evals-retrieval`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on for is are was were be by with at as from this that it its".split())


_PERIOD = re.compile(r"(\d{4})-(\d{2})\.md$")


@dataclass(frozen=True)
class Chunk:
    id: str             # source#n, stable across runs
    source: str         # corpus-relative path
    section: str        # nearest heading
    text: str           # title + heading + paragraph, the text that is searched and shown
    period: str | None = None  # "2026-07" for monthly commentary, None for undated docs

    @property
    def body(self) -> str:
        """The paragraph without the title and heading prefix."""
        return self.text.split("\n", 2)[-1]


def chunk_markdown(path: Path, root: Path = CORPUS) -> list[Chunk]:
    """One chunk per paragraph; the document title and heading are prefixed for context."""
    rel = path.relative_to(root).as_posix()
    m = _PERIOD.search(rel)
    period = f"{m.group(1)}-{m.group(2)}" if m and rel.startswith("commentary/") else None
    title, section, chunks = "", "", []
    para: list[str] = []

    def flush():
        body = " ".join(para).strip()
        para.clear()
        if body:
            chunks.append(Chunk(f"{rel}#{len(chunks)}", rel, section or title,
                                f"{title}\n{section or title}\n{body}", period))

    for line in path.read_text().splitlines():
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return chunks


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_corpus(cls, root: Path = CORPUS) -> "BM25Index":
        chunks = [c for p in sorted(root.rglob("*.md")) for c in chunk_markdown(p, root)]
        return cls(chunks)

    def search(self, query: str, k: int = 5, periods: set[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks by BM25. With `periods`, dated chunks outside them are skipped;
        undated documents (policy, memos, notes) are always eligible."""
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lens):
            if periods is not None and chunk.period is not None and chunk.period not in periods:
                continue
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            if s > 0:
                scored.append((chunk, s))
        return sorted(scored, key=lambda x: -x[1])[:k]
