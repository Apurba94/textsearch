"""The on-disk index: immutable segments of compressed postings, plus a live-document list.

New documents collect in memory and flush to a new segment. A segment never changes;
deletes are recorded separately, and merging segments rewrites them and drops the dead.
Each postings list is doc-id gaps and position gaps, stored as variable-length integers.
"""
import json
import math
import os
import struct

from .analyze import Analyzer


def vbyte(numbers):
    out = bytearray()
    for n in numbers:
        while n >= 128:
            out.append((n & 127) | 128)
            n >>= 7
        out.append(n)
    return bytes(out)


def unvbyte(data):
    out, n, shift = [], 0, 0
    for b in data:
        n |= (b & 127) << shift
        if b & 128:
            shift += 7
        else:
            out.append(n)
            n, shift = 0, 0
    return out


def encode_postings(postings):
    """postings: sorted list of (doc, [positions])."""
    nums, last = [], 0
    for doc, positions in postings:
        nums += [doc - last, len(positions)]
        last = doc
        prev = 0
        for p in positions:
            nums.append(p - prev)
            prev = p
    return vbyte(nums)


def decode_postings(data):
    nums = unvbyte(data)
    out, i, doc = [], 0, 0
    while i < len(nums):
        doc += nums[i]
        count = nums[i + 1]
        i += 2
        positions, p = [], 0
        for k in range(count):
            p += nums[i + k]
            positions.append(p)
        i += count
        out.append((doc, positions))
    return out


class Segment:
    """One immutable segment. `base` turns its local numbers into index-wide document ids."""

    def __init__(self, path, name):
        self.name = name
        self.path = os.path.join(path, name)
        with open(self.path + ".terms", encoding="utf-8") as f:
            self.terms = json.load(f)        # {field: {term: [offset, length, doc freq]}}
        with open(self.path + ".docs", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.post = open(self.path + ".post", "rb")

    def postings(self, field, term):
        entry = self.terms.get(field, {}).get(term)
        if entry is None:
            return []
        self.post.seek(entry[0])
        return decode_postings(self.post.read(entry[1]))

    def vocabulary(self, field):
        return self.terms.get(field, {})

    def close(self):
        self.post.close()


def write_segment(path, name, docs):
    """docs: list of {"id", "fields", "lengths", "terms": {field: {term: [positions]}}}."""
    post = open(os.path.join(path, name + ".post"), "wb")
    table = {}
    grouped = {}
    for number, doc in enumerate(docs):
        for field, terms in doc["terms"].items():
            for term, positions in terms.items():
                grouped.setdefault(field, {}).setdefault(term, []).append((number, positions))
    offset = 0
    for field in sorted(grouped):
        table[field] = {}
        for term in sorted(grouped[field]):
            data = encode_postings(grouped[field][term])
            post.write(data)
            table[field][term] = [offset, len(data), len(grouped[field][term])]
            offset += len(data)
    post.close()
    with open(os.path.join(path, name + ".terms"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    with open(os.path.join(path, name + ".docs"), "w", encoding="utf-8") as f:
        json.dump([{"id": d["id"], "fields": d["fields"], "lengths": d["lengths"]} for d in docs], f, ensure_ascii=False)


class Index:
    """A search index in a folder. Changes become visible and durable at `commit()`."""

    def __init__(self, path, analyzer=None, flush_at=2000):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.meta_path = os.path.join(path, "meta.json")
        self.flush_at = flush_at
        self.buffer = []
        self.pending_deletes = set()
        if os.path.exists(self.meta_path):
            with open(self.meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.analyzer = Analyzer(**meta["analyzer"])
            self.names, self.deleted, self.next_seg = meta["segments"], {k: set(v) for k, v in meta["deleted"].items()}, meta["next"]
        else:
            self.analyzer = analyzer or Analyzer()
            self.names, self.deleted, self.next_seg = [], {}, 0
            self._save()
        self.segments = [Segment(path, n) for n in self.names]
        self._stats()

    def _save(self):
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"analyzer": self.analyzer.config(), "segments": self.names, "deleted": {k: sorted(v) for k, v in self.deleted.items()}, "next": self.next_seg}, f)
        os.replace(tmp, self.meta_path)

    def _stats(self):
        self.count = 0
        self.total = {}
        self.ids = {}
        for seg in self.segments:
            dead = self.deleted.get(seg.name, ())
            for i, d in enumerate(seg.docs):
                if i not in dead:
                    self.count += 1
                    self.ids[d["id"]] = (seg, i)
                    for field, n in d["lengths"].items():
                        self.total[field] = self.total.get(field, 0) + n

    def add(self, doc_id, **fields):
        """Add a document, replacing any with the same id. Field values are text."""
        doc_id = str(doc_id)
        self.pending_deletes.add(doc_id)
        self.buffer = [d for d in self.buffer if d["id"] != doc_id]
        terms, lengths = {}, {}
        for field, text in fields.items():
            text = str(text)
            per = {}
            count = 0
            for term, pos, _, _ in self.analyzer.tokens(text):
                per.setdefault(term, []).append(pos)
                count += 1
            terms[field], lengths[field] = per, count
        self.buffer.append({"id": doc_id, "fields": {k: str(v) for k, v in fields.items()}, "lengths": lengths, "terms": terms})
        if len(self.buffer) >= self.flush_at:
            self._flush()

    def delete(self, doc_id):
        doc_id = str(doc_id)
        self.pending_deletes.add(doc_id)
        self.buffer = [d for d in self.buffer if d["id"] != doc_id]

    def _flush(self):
        fresh = {d["id"] for d in self.buffer}
        for doc_id in self.pending_deletes:
            hit = self.ids.get(doc_id)
            if hit:
                self.deleted.setdefault(hit[0].name, set()).add(hit[1])
        self.pending_deletes = set()
        if self.buffer:
            name = f"seg{self.next_seg}"
            self.next_seg += 1
            write_segment(self.path, name, self.buffer)
            self.names.append(name)
            self.segments.append(Segment(self.path, name))
            self.buffer = []
        return fresh

    def commit(self):
        self._flush()
        self._save()
        self._stats()

    def merge(self):
        """Combine all segments into one and drop deleted documents."""
        self.commit()
        docs = []
        for seg in self.segments:
            dead = self.deleted.get(seg.name, ())
            local = {}
            for field, table in seg.terms.items():
                for term in table:
                    for number, positions in seg.postings(field, term):
                        if number not in dead:
                            local.setdefault(number, {}).setdefault(field, {})[term] = positions
            for i, d in enumerate(seg.docs):
                if i not in dead:
                    docs.append({"id": d["id"], "fields": d["fields"], "lengths": d["lengths"], "terms": local.get(i, {})})
        old = self.names
        for seg in self.segments:
            seg.close()
        name = f"seg{self.next_seg}"
        self.next_seg += 1
        write_segment(self.path, name, docs)
        self.names, self.deleted = [name], {}
        self._save()
        for n in old:
            for ext in (".terms", ".docs", ".post"):
                os.remove(os.path.join(self.path, n + ext))
        self.segments = [Segment(self.path, name)]
        self._stats()

    def doc_freq(self, field, term):
        n = 0
        for seg in self.segments:
            dead = self.deleted.get(seg.name, ())
            for number, _ in seg.postings(field, term):
                n += number not in dead
        return n

    def average_length(self, field):
        return self.total.get(field, 0) / self.count if self.count else 0

    def vocabulary(self, field):
        out = set()
        for seg in self.segments:
            out |= set(seg.vocabulary(field))
        return out

    def fields(self):
        out = set()
        for seg in self.segments:
            out |= set(seg.terms)
        return out

    def get(self, doc_id):
        hit = self.ids.get(str(doc_id))
        return None if hit is None else hit[0].docs[hit[1]]["fields"]

    def close(self):
        for seg in self.segments:
            seg.close()
