import math
import sys

import requests

URL = "http://localhost:11434/api/embed"


def embed(texts):
    r = requests.post(URL, json={"model": "nomic-embed-text", "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


snippets = [
    "def run_tests(self, path=''): runs pytest if installed, otherwise unittest discover",
    "def read_file(self, path, start_line=1): returns the file with line numbers",
    "class LRUCache: evicts the least recently used key when full",
    "def slugify(title): lowercase, replace non-alphanumeric runs with a dash",
    "sys.exit('Ollama is not running. Open the Ollama app and try again.')",
]
query = sys.argv[1] if len(sys.argv) > 1 else "Which function executes the test suite?"

# nomic-embed-text works best when told which side is the question and which the documents
vectors = embed(["search_query: " + query] + ["search_document: " + s for s in snippets])
q, docs = vectors[0], vectors[1:]

print(f"Each text became {len(q)} numbers. First five: {[round(x, 3) for x in q[:5]]}\n")
print(f"Query: {query}\n")
for score, s in sorted(((cosine(q, d), s) for d, s in zip(docs, snippets)), reverse=True):
    print(f"{score:.3f}  {s}")