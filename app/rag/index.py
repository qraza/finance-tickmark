"""Document retrieval over the corpus, with citable chunks.

Chunks are heading sections, split into paragraphs only when long, so one monthly
commentary yields a separate passage for each variance it explains while a short
metric definition stays in one piece. Commentary
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


SHORT_SECTION = 600  # chars: sections up to this size stay whole, longer ones split by paragraph


def chunk_markdown(path: Path, root: Path = CORPUS) -> list[Chunk]:
    """Chunk by heading section; split a section into paragraphs only when it's long.

    Short sections (a metric definition, an owner note) stay whole so a definition and
    its formula are retrieved together. Long ones (a month's variance notes) split per
    paragraph so each variance is its own citable passage. The document title and
    heading are prefixed to every chunk for context.
    """
    rel = path.relative_to(root).as_posix()
    m = _PERIOD.search(rel)
    period = f"{m.group(1)}-{m.group(2)}" if m and rel.startswith("commentary/") else None
    title, section, chunks = "", "", []
    paras: list[str] = []
    para: list[str] = []

    def end_para():
        if para:
            paras.append(" ".join(para).strip())
            para.clear()

    def end_section():
        end_para()
        if not paras:
            return
        name = section or title
        groups = ["\n".join(paras)] if sum(map(len, paras)) <= SHORT_SECTION else list(paras)
        for body in groups:
            chunks.append(Chunk(f"{rel}#{len(chunks)}", rel, name, f"{title}\n{name}\n{body}", period))
        paras.clear()

    for line in path.read_text().splitlines():
        if line.startswith("# "):
            end_section()
            title = line[2:].strip()
        elif line.startswith("## "):
            end_section()
            section = line[3:].strip()
        elif not line.strip():
            end_para()
        else:
            para.append(line.strip())
    end_section()
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

    def search(self, query: str, k: int = 5, periods: set[str] | None = None,
               exclude: tuple[str, ...] = ()) -> list[tuple[Chunk, float]]:
        """Top-k chunks by BM25. With `periods`, dated chunks outside them are skipped;
        undated documents (policy, memos, notes) are always eligible. `exclude` drops
        sources by path prefix."""
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lens):
            if periods is not None and chunk.period is not None and chunk.period not in periods:
                continue
            if exclude and chunk.source.startswith(exclude):
                continue
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            if s > 0:
                scored.append((chunk, s))
        return sorted(scored, key=lambda x: -x[1])[:k]
