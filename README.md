# TextSearch

A full-text search engine written from scratch in pure Python, built the way real engines such as Lucene work: immutable segments, compressed postings and BM25 ranking.

```python
from textsearch import Index, Searcher

ix = Index("my-index")
ix.add("doc1", title="Water supply", body="Clean water points reduce disease")
ix.add("doc2", title="Shelter repairs", body="Families repair shelters before the rains")
ix.commit()

s = Searcher(ix, boosts={"title": 2.0})
for hit in s.search('water AND "clean water"'):
    print(hit["id"], round(hit["score"], 3), s.snippet(hit["fields"]["body"], "water"))
```

Or index a folder of text files:

```
python -m textsearch index my-index ./notes
python -m textsearch search my-index "water quality"
```

## Query language

| Query | Meaning |
|---|---|
| `water quality` | Either word, ranked by relevance |
| `water AND quality`, `water OR food`, `water NOT quality` | Boolean logic, with parentheses |
| `+water -testing` | Must have and must not have |
| `"water supply"` | Exact phrase |
| `title:shelter` | Search one field |
| `settle*` | Prefix |
| `watr~` | Fuzzy, tolerating typing mistakes |

## How it works

- **Analysis:** text becomes lower-case tokens, with a Porter-style stemmer for English. Bangla and other scripts are kept as whole words. Stopwords are dropped but keep their position, so phrases stay accurate.
- **Segments:** documents are written to immutable segments. Deletes are recorded beside them, and `merge()` rewrites everything without the dead documents.
- **Compression:** postings store gaps between document numbers and positions as variable-length integers.
- **Ranking:** BM25 with field boosts, checked in the tests against a calculation by hand.

## How it is tested

200 random boolean and phrase queries are compared with a plain scan of the documents, across several segments and after deletes. Merging must keep the same results and the same scores.

## Limits

Term tables are loaded into memory when an index opens, so this suits collections of up to about a few hundred thousand short documents. Only one process should write at a time. There is no stemmer for Bangla.

## Tests

```
python -m unittest discover -s tests
```

MIT licence.
