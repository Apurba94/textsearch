import math
import os
import random
import shutil
import tempfile
import unittest

from textsearch import Analyzer, Index, QueryError, Searcher, decode_postings, edit_distance, encode_postings, stem

DOCS = {
    "1": ("Water supply in camps", "Clean water points reduce disease in camps and settlements"),
    "2": ("Teaching digital skills", "Teachers run computer classes for young learners in the camp"),
    "3": ("Food distribution", "Food baskets are delivered every month to registered families"),
    "4": ("Water quality testing", "Testing water quality each week keeps the water supply safe"),
    "5": ("Shelter repairs", "Families repair shelters before the monsoon rains"),
}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ix")
        self.ix = Index(self.path)
        for i, (t, b) in DOCS.items():
            self.ix.add(i, title=t, body=b)
        self.ix.commit()
        self.s = Searcher(self.ix)

    def tearDown(self):
        self.ix.close()
        shutil.rmtree(self.dir)

    def ids(self, q):
        return sorted(h["id"] for h in self.s.search(q, 100))


class TestAnalysis(unittest.TestCase):
    def test_stemmer(self):
        for a, b in [("caresses", "caress"), ("ponies", "poni"), ("running", "run"), ("agreed", "agre"), ("tests", "test"), ("testing", "test"), ("hopping", "hop"), ("relational", "relat")]:
            self.assertEqual(stem(a), b, a)

    def test_related_forms_meet(self):
        self.assertEqual(stem("connection"), stem("connections"))
        self.assertEqual(stem("teachers"), stem("teacher"))

    def test_bangla_tokens_survive(self):
        self.assertEqual(Analyzer().terms("পানি সরবরাহ"), ["পানি", "সরবরাহ"])

    def test_stopwords_keep_their_positions(self):
        self.assertEqual([(t, p) for t, p, _, _ in Analyzer().tokens("the cat in the hat")], [("cat", 1), ("hat", 4)])

    def test_postings_round_trip(self):
        rng = random.Random(1)
        for _ in range(100):
            doc, p = 0, []
            for _ in range(rng.randint(0, 30)):
                doc += rng.randint(1, 100000)
                p.append((doc, sorted(rng.sample(range(5000), rng.randint(1, 8)))))
            self.assertEqual(decode_postings(encode_postings(p)), p)

    def test_edit_distance(self):
        self.assertEqual(edit_distance("water", "water", 2), 0)
        self.assertEqual(edit_distance("water", "wtaer", 2), 1)
        self.assertEqual(edit_distance("water", "wate", 2), 1)
        self.assertEqual(edit_distance("water", "fire", 1), 2)


class TestQueries(Base):
    def test_single_term_with_stemming(self):
        self.assertEqual(self.ids("repair"), ["5"])
        self.assertEqual(self.ids("teachers"), ["2"])

    def test_or_is_the_default(self):
        self.assertEqual(self.ids("shelter food"), ["3", "5"])

    def test_and_or_not(self):
        self.assertEqual(self.ids("water AND quality"), ["4"])
        self.assertEqual(self.ids("water NOT quality"), ["1"])
        self.assertEqual(self.ids("(food OR shelter) AND families"), ["3", "5"])

    def test_plus_and_minus(self):
        self.assertEqual(self.ids("+water -testing"), ["1"])
        self.assertEqual(self.ids("+camp computer"), ["1", "2"])

    def test_pure_negative(self):
        self.assertEqual(self.ids("-water"), ["2", "3", "5"])

    def test_phrases(self):
        self.assertEqual(self.ids('"water supply"'), ["1", "4"])
        self.assertEqual(self.ids('"supply water"'), [])
        self.assertEqual(self.ids('"water points reduce"'), ["1"])

    def test_phrase_across_a_stopword_needs_the_gap(self):
        self.ix.add("6", body="river of life")
        self.ix.add("7", body="river life")
        self.ix.commit()
        self.assertEqual(self.ids('"river of life"'), ["6"])

    def test_field_query(self):
        self.assertEqual(self.ids("title:shelter"), ["5"])
        self.assertEqual(self.ids("title:families"), [])

    def test_prefix_and_fuzzy(self):
        self.assertEqual(self.ids("settle*"), ["1"])
        self.assertEqual(self.ids("watr~"), ["1", "4"])
        self.assertEqual(self.ids("shelters~"), ["5"])

    def test_empty_and_bad(self):
        self.assertEqual(self.s.search(""), [])
        self.assertEqual(self.s.search("the of"), [])
        with self.assertRaises(QueryError):
            self.s.search("(water")
        with self.assertRaises(QueryError):
            self.s.search("water)")

    def test_snippet_marks_the_match(self):
        text = "x " * 200 + "the water points are clean " + "y " * 200
        s = self.s.snippet(text, "water points")
        self.assertIn("[water]", s)
        self.assertIn("[points]", s)
        self.assertLess(len(s), 200)


class TestRanking(Base):
    def test_more_matches_rank_higher(self):
        self.assertEqual(self.s.search("water")[0]["id"], "4")

    def test_boost_lifts_title_hits(self):
        ix = self.ix
        ix.add("a", title="monsoon", body="other words here")
        ix.add("b", title="other", body="monsoon words here")
        ix.commit()
        hits = [h["id"] for h in Searcher(ix, boosts={"title": 5.0}).search("monsoon")]
        self.assertEqual(hits[0], "a")

    def test_bm25_matches_a_hand_calculation(self):
        words = [["a1", "b1"], ["a1", "a1", "c1"], ["c1", "d1", "e1", "f1"]]
        d = os.path.join(self.dir, "b")
        ix = Index(d, Analyzer(stemming=False, stopwords=False))
        for i, w in enumerate(words):
            ix.add(i, body=" ".join(w))
        ix.commit()
        s = Searcher(ix)
        got = {h["id"]: h["score"] for h in s.search("a1")}
        N, n, avg = 3, 2, 3.0
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
        def score(tf, dl):
            return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avg))
        self.assertAlmostEqual(got["0"], score(1, 2))
        self.assertAlmostEqual(got["1"], score(2, 3))
        ix.close()


class TestLifecycle(Base):
    def test_survives_reopen(self):
        self.ix.close()
        self.ix = Index(self.path)
        self.s = Searcher(self.ix)
        self.assertEqual(self.ids("repair"), ["5"])

    def test_uncommitted_is_invisible_and_lost(self):
        self.ix.add("9", body="zebra")
        self.assertEqual(self.ids("zebra"), [])
        self.ix.close()
        self.ix = Index(self.path)
        self.s = Searcher(self.ix)
        self.assertEqual(self.ids("zebra"), [])

    def test_delete(self):
        self.ix.delete("5")
        self.ix.commit()
        self.assertEqual(self.ids("repair"), [])
        self.assertEqual(self.ix.count, 4)
        self.assertIsNone(self.ix.get("5"))

    def test_add_replaces_same_id(self):
        self.ix.add("5", body="now about bicycles")
        self.ix.commit()
        self.assertEqual(self.ids("repair"), [])
        self.assertEqual(self.ids("bicycle"), ["5"])
        self.assertEqual(self.ix.count, 5)

    def test_merge_keeps_results_and_drops_files(self):
        self.ix.add("6", body="more water")
        self.ix.commit()
        self.ix.delete("1")
        self.ix.commit()
        before = self.s.search("water", 100)
        self.ix.merge()
        self.assertEqual(len(self.ix.segments), 1)
        after = self.s.search("water", 100)
        self.assertEqual([h["id"] for h in after], [h["id"] for h in before])
        for a, b in zip(after, before):
            self.assertAlmostEqual(a["score"], b["score"])
        self.assertEqual(sum(f.endswith(".post") for f in os.listdir(self.path)), 1)

    def test_flush_threshold_makes_segments(self):
        d = os.path.join(self.dir, "f")
        ix = Index(d, flush_at=10)
        for i in range(35):
            ix.add(i, body=f"word{i} common")
        ix.commit()
        self.assertEqual(len(ix.segments), 4)
        self.assertEqual(Searcher(ix).count("common"), 35)
        ix.close()


class TestAgainstBruteForce(unittest.TestCase):
    """Random boolean queries must match a plain scan of the documents."""

    def test_random(self):
        rng = random.Random(5)
        d = tempfile.mkdtemp()
        ix = Index(os.path.join(d, "ix"), Analyzer(stemming=False, stopwords=False), flush_at=40)
        vocab = [f"w{i}" for i in range(30)]
        docs = {}
        for i in range(200):
            words = [rng.choice(vocab) for _ in range(rng.randint(1, 25))]
            docs[str(i)] = words
            ix.add(i, body=" ".join(words))
            if i % 50 == 49:
                ix.commit()
        ix.commit()
        for i in rng.sample(range(200), 30):
            ix.delete(i)
            del docs[str(i)]
        ix.commit()
        s = Searcher(ix)
        for _ in range(200):
            a, b, c = rng.sample(vocab, 3)
            kind = rng.randrange(5)
            if kind == 0:
                q, want = f"{a} OR {b}", lambda w: a in w or b in w
            elif kind == 1:
                q, want = f"{a} AND {b}", lambda w: a in w and b in w
            elif kind == 2:
                q, want = f"{a} AND NOT {b}", lambda w: a in w and b not in w
            elif kind == 3:
                q, want = f"({a} OR {b}) AND {c}", lambda w: (a in w or b in w) and c in w
            else:
                q, want = f'"{a} {b}"', lambda w: any(x == a and y == b for x, y in zip(w, w[1:]))
            expected = sorted(i for i, w in docs.items() if want(w))
            self.assertEqual(sorted(h["id"] for h in s.search(q, 1000)), expected, q)
        ix.close()
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
