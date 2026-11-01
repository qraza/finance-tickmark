"""Document retrieval over the corpus, with citable chunks.

Baseline: heading-aware markdown chunking + BM25, pure Python, no dependencies.
Week 3 of the build: add embeddings (DuckDB `vss` or LanceDB) and fuse with BM25
(reciprocal rank fusion), then measure whether it beats this baseline on the golden set.
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


@dataclass(frozen=True)
class Chunk:
    source: str      # corpus-relative path
    section: str     # nearest heading
    text: str


def chunk_markdown(path: Path, root: Path = CORPUS) -> list[Chunk]:
    """One chunk per heading section; the document title is prefixed for context."""
    rel = path.relative_to(root).as_posix()
    title, section, buf, chunks = "", "", [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(rel, section or title, f"{title}\n{section}\n{body}".strip()))

    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        else:
            buf.append(line)
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

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            if s > 0:
                scored.append((chunk, s))
        return sorted(scored, key=lambda x: -x[1])[:k]
