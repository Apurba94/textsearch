"""Query parsing, boolean matching, BM25 ranking, fuzzy terms and snippets.

Query language: words (any may match), +word (must match), -word (must not), AND, OR, NOT,
"exact phrase", title:word, pre*, word~ (fuzzy), and parentheses.
"""
import math
import re
from dataclasses import dataclass

TOKEN = re.compile(r'\s*(?:(?P<paren>[()])|(?P<sign>[+-])(?=\S)|(?:(?P<field>[^\W\d_][\w]*):)?(?:"(?P<phrase>[^"]*)"|(?P<word>[^\s()"]+)))', re.UNICODE)


class QueryError(Exception):
    pass


@dataclass
class Term:
    field: str
    text: str
    kind: str = "term"    # term, prefix or fuzzy


@dataclass
class Phrase:
    field: str
    terms: list


@dataclass
class And:
    items: list


@dataclass
class Or:
    items: list


@dataclass
class Not:
    item: object


@dataclass
class All:
    pass


def lex(q):
    pos, out = 0, []
    while pos < len(q):
        if q[pos:].strip() == "":
            break
        m = TOKEN.match(q, pos)
        if not m or m.end() == pos:
            raise QueryError(f"cannot read the query at: {q[pos:]}")
        pos = m.end()
        if m.group("paren"):
            out.append(("paren", m.group("paren")))
        elif m.group("sign"):
            out.append(("sign", m.group("sign")))
        elif m.group("phrase") is not None:
            out.append(("phrase", m.group("field"), m.group("phrase")))
        else:
            w = m.group("word")
            if w in ("AND", "OR", "NOT") and not m.group("field"):
                out.append(("op", w))
            else:
                out.append(("word", m.group("field"), w))
    return out


class Parser:
    def __init__(self, index, q, fields):
        self.ix, self.t, self.i, self.fields = index, lex(q), 0, fields

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def parse(self):
        if not self.t:
            return Or([])
        e = self.or_expr()
        if self.i < len(self.t):
            raise QueryError("unbalanced parenthesis")
        return e

    def or_expr(self):
        items = [self.and_expr()]
        while self.peek() == ("op", "OR"):
            self.i += 1
            items.append(self.and_expr())
        return items[0] if len(items) == 1 else Or(items)

    def and_expr(self):
        """Adjacent words are optional unless AND, + or - says otherwise."""
        must, should, mustnot = [], [], []
        pending_and = False
        count = 0
        while True:
            kind = self.peek()[0]
            if kind is None or self.peek() == ("paren", ")") or self.peek() == ("op", "OR"):
                break
            if self.peek() == ("op", "AND"):
                self.i += 1
                pending_and = True
                if should:
                    must.append(should.pop())
                continue
            sign = None
            if kind == "sign":
                sign = self.t[self.i][1]
                self.i += 1
            if self.peek() == ("op", "NOT"):
                self.i += 1
                sign = "-"
            item = self.atom()
            count += 1
            if item is None:
                continue
            if sign == "-":
                mustnot.append(item)
            elif sign == "+" or pending_and:
                must.append(item)
            else:
                should.append(item)
            pending_and = False
        if count == 0:
            raise QueryError("expected a search term")
        if not must and not mustnot:
            return should[0] if len(should) == 1 else Or(should)
        if not must and not should:
            return And([All()] + [Not(x) for x in mustnot])
        out = must + [Not(x) for x in mustnot]
        if should:
            return And(([Or(should)] if not must else out[:len(must)]) + [Not(x) for x in mustnot]) if not must else _with_optional(out, should)
        return And(out)

    def atom(self):
        tok = self.peek()
        if tok == ("paren", "("):
            self.i += 1
            e = self.or_expr()
            if self.peek() != ("paren", ")"):
                raise QueryError("missing )")
            self.i += 1
            return e
        if tok[0] not in ("word", "phrase"):
            raise QueryError(f"unexpected {tok[1]}")
        self.i += 1
        fields = [tok[1]] if tok[1] else self.fields
        if tok[0] == "phrase":
            terms = list(self.ix.analyzer.tokens(tok[2]))
            if not terms:
                return None
            base = terms[0][1]
            built = [Phrase(f, [(t[0], t[1] - base) for t in terms]) for f in fields]
            return built[0] if len(built) == 1 else Or(built)
        word = tok[2]
        if word.endswith("*") and len(word) > 1:
            made = [Term(f, word[:-1].lower(), "prefix") for f in fields]
        elif word.endswith("~") and len(word) > 1:
            made = [Term(f, word[:-1].lower(), "fuzzy") for f in fields]
        else:
            terms = self.ix.analyzer.terms(word)
            if not terms:
                return None
            made = [Term(f, terms[0]) for f in fields]
        return made[0] if len(made) == 1 else Or(made)


def _with_optional(items, should):
    node = And(items)
    node.optional = should
    return node


def edit_distance(a, b, limit):
    """Damerau-Levenshtein with an early exit once every cell exceeds the limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cost = x != y
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and x == b[j - 2] and a[i - 2] == y:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class Searcher:
    k1 = 1.2
    b = 0.75

    def __init__(self, index, fields=None, boosts=None):
        self.ix = index
        self.fields = fields or sorted(index.fields())
        self.boosts = boosts or {}

    def expand(self, term):
        vocab = self.ix.vocabulary(term.field)
        if term.kind == "prefix":
            return sorted(t for t in vocab if t.startswith(term.text))
        if term.kind == "fuzzy":
            limit = 1 if len(term.text) <= 5 else 2
            return sorted(t for t in vocab if edit_distance(term.text, t, limit) <= limit)
        return [term.text]

    def term_docs(self, field, term):
        """{(segment, number): positions} for a term, without deleted documents."""
        out = {}
        for seg in self.ix.segments:
            dead = self.ix.deleted.get(seg.name, ())
            for number, positions in seg.postings(field, term):
                if number not in dead:
                    out[(seg.name, number)] = positions
        return out

    def idf(self, field, term):
        n = self.ix.doc_freq(field, term)
        return math.log(1 + (self.ix.count - n + 0.5) / (n + 0.5))

    def score_term(self, field, term, docs, scores):
        idf = self.idf(field, term)
        avg = self.ix.average_length(field) or 1
        boost = self.boosts.get(field, 1.0)
        length = {s.name: s for s in self.ix.segments}
        for key, positions in docs.items():
            seg, number = key
            n = length[seg].docs[number]["lengths"].get(field, 0)
            tf = len(positions)
            scores[key] = scores.get(key, 0.0) + boost * idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * n / avg))

    def run(self, node):
        """Return {doc key: score} for the documents that match."""
        if isinstance(node, Term):
            scores = {}
            for t in self.expand(node):
                self.score_term(node.field, t, self.term_docs(node.field, t), scores)
            return scores
        if isinstance(node, Phrase):
            lists = [self.term_docs(node.field, t) for t, _ in node.terms]
            keys = set(lists[0])
            for docs in lists[1:]:
                keys &= set(docs)
            hits = {}
            for key in keys:
                starts = {p - node.terms[0][1] for p in lists[0][key]}
                for docs, (_, gap) in zip(lists[1:], node.terms[1:]):
                    starts &= {p - gap for p in docs[key]}
                if starts:
                    hits[key] = [0] * len(starts)
            scores = {}
            for t, _ in node.terms:
                self.score_term(node.field, t, hits, scores)
            return scores
        if isinstance(node, All):
            return {(s.name, i): 0.0 for s in self.ix.segments for i in range(len(s.docs)) if i not in self.ix.deleted.get(s.name, ())}
        if isinstance(node, Or):
            scores = {}
            for item in node.items:
                for k, v in self.run(item).items():
                    scores[k] = scores.get(k, 0.0) + v
            return scores
        if isinstance(node, And):
            scores = None
            excluded = set()
            for item in node.items:
                if isinstance(item, Not):
                    excluded |= set(self.run(item.item))
                    continue
                r = self.run(item)
                scores = r if scores is None else {k: scores[k] + r[k] for k in scores.keys() & r.keys()}
            scores = scores or {}
            for k in excluded:
                scores.pop(k, None)
            for item in getattr(node, "optional", ()):
                for k, v in self.run(item).items():
                    if k in scores:
                        scores[k] += v
            return scores
        raise QueryError(f"cannot run {node!r}")

    def search(self, query, limit=10):
        """Return hits as dicts: id, score and fields, best first."""
        node = Parser(self.ix, query, self.fields).parse()
        scores = self.run(node)
        segs = {s.name: s for s in self.ix.segments}
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], segs[kv[0][0]].docs[kv[0][1]]["id"]))
        out = []
        for (seg, number), score in ranked[:limit]:
            d = segs[seg].docs[number]
            out.append({"id": d["id"], "score": score, "fields": d["fields"]})
        return out

    def count(self, query):
        return len(self.run(Parser(self.ix, query, self.fields).parse()))

    def snippet(self, text, query, size=160, mark=("[", "]")):
        """The window of text with the most query terms, with matches marked."""
        wanted = set()

        def collect(n):
            if isinstance(n, Term):
                wanted.update(self.expand(n))
            elif isinstance(n, Phrase):
                wanted.update(t for t, _ in n.terms)
            elif isinstance(n, (And, Or)):
                for x in n.items:
                    collect(x)
                for x in getattr(n, "optional", ()):
                    collect(x)
        collect(Parser(self.ix, query, self.fields).parse())
        hits = [(s, e) for term, _, s, e in self.ix.analyzer.tokens(text) if term in wanted]
        if not hits:
            return text[:size]
        best, best_count = 0, 0
        for i, (s, _) in enumerate(hits):
            count = sum(1 for a, _ in hits[i:] if a < s + size)
            if count > best_count:
                best, best_count = i, count
        start = max(0, hits[best][0] - 20)
        end = min(len(text), start + size)
        out, pos = [], start
        for s, e in hits:
            if s >= start and e <= end:
                out += [text[pos:s], mark[0], text[s:e], mark[1]]
                pos = e
        out.append(text[pos:end])
        return ("…" if start else "") + "".join(out) + ("…" if end < len(text) else "")
