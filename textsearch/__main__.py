"""Command line: python -m textsearch index INDEX FOLDER, or search INDEX "QUERY"."""
import os
import sys

from . import Index, Searcher


def main(argv):
    if len(argv) >= 4 and argv[1] == "index":
        ix = Index(argv[2])
        n = 0
        for base, _, files in os.walk(argv[3]):
            for name in files:
                if name.endswith((".txt", ".md")):
                    path = os.path.join(base, name)
                    with open(path, encoding="utf-8", errors="replace") as f:
                        ix.add(path, title=name, body=f.read())
                    n += 1
        ix.commit()
        print(f"indexed {n} files, {ix.count} documents in total")
        ix.close()
        return 0
    if len(argv) >= 4 and argv[1] == "search":
        ix = Index(argv[2])
        s = Searcher(ix, boosts={"title": 2.0})
        query = " ".join(argv[3:])
        for hit in s.search(query):
            print(f"{hit['score']:7.3f}  {hit['id']}")
            print("         " + s.snippet(hit["fields"].get("body", ""), query).replace("\n", " "))
        ix.close()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
