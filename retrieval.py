"""
retrieval.py - keyword serach over the notes/ folder; how the assistant cites its method.

same shape as a vector db: turn text into vectors -> store them -> return the nearest ones to a query.

here the vectors are tf-idf weights and similarity is the cosine, which is plenty for a few hundred paragraphs.
swapping embed() for an embedding model and the dict for a vector store changes nothing above this file.

    python retrieval.py "what does months of supply mean"

"""

import glob
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(HERE, "notes")
TOKEN = re.compile(r"[a-z0-9$%]+")
STOP = set("a an the and or of to in on for is are was be by with as at it this that from vs into than "
           "we you your our their its not no if so do does can will".split())


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def load_passages(folder=NOTES_DIR):
    """one passage per paragraph, tagged with file and paragraph index so answers can cite."""
    passages = []
    for path in sorted(glob.glob(os.path.join(folder, "*.md"))):
        with open(path, encoding="utf-8") as fh:
            paras = [p.strip() for p in fh.read().split("\n\n") if p.strip()]
        title = paras[0].lstrip("# ").strip() if paras and paras[0].startswith("#") else os.path.basename(path)
        for i, p in enumerate(paras):
            if p.startswith("#"):
                continue
            passages.append({"id": f"{os.path.basename(path)}#{i}", "source": os.path.basename(path),
                             "title": title, "text": p})
    return passages


class Index:
    def __init__(self, passages):
        self.passages = passages
        self.n = len(passages)
        self.df = Counter()
        self.vectors = []
        for p in passages:
            tf = Counter(tokenize(p["text"] + " " + p["title"]))
            self.df.update(tf.keys())
            self.vectors.append(tf)
        # idf with the +1 smoothing so a term in every passage still counts a little
        self.idf = {t: math.log((1 + self.n) / (1 + d)) + 1.0 for t, d in self.df.items()}
        self.vectors = [self._weigh(tf) for tf in self.vectors]

    def _weigh(self, tf):
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def embed(self, text):
        return self._weigh(Counter(tokenize(text)))

    def search(self, query, k=3):
        q = self.embed(query)
        scored = []
        for p, v in zip(self.passages, self.vectors):
            # cosine: both vectors are unit length, so the dot product is the similarity
            score = sum(w * v.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda s: -s[0])
        return [{"score": round(s, 3), **p} for s, p in scored[:k]]


_index = None


def search_notes(query, k=3):
    """tool-shaped entry point: returns passages the agent (or a web app) can quote with a source."""
    global _index
    if _index is None:
        _index = Index(load_passages())
    hits = _index.search(query, k)
    if not hits:
        return {"query": query, "hits": [], "note": "nothing relevant in the notes"}
    return {"query": query, "hits": hits}


if __name__ == "__main__":
    import sys
    for h in search_notes(" ".join(sys.argv[1:]) or "months of supply")["hits"]:
        print(f"{h['score']:.3f}  {h['id']}\n    {h['text'][:160]}...")
