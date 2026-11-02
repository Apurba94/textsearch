"""Text analysis: split text into tokens and reduce words to a common form."""
import re
import unicodedata

STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with".split())
VOWEL = "aeiou"


def words(text):
    """Yield (word, start, end). Letters, digits and combining marks all count, because
    a regex word match skips the vowel signs of Bangla and other Indic scripts and would split words."""
    i, n = 0, len(text)
    while i < n:
        if unicodedata.category(text[i])[0] not in "LN":
            i += 1
            continue
        start = i
        while i < n and (unicodedata.category(text[i])[0] in "LNM" or (text[i] == "'" and i + 1 < n and unicodedata.category(text[i + 1])[0] in "LN")):
            i += 1
        yield text[start:i], start, i


def _measure(stem):
    """Porter's m: the number of vowel-then-consonant sequences."""
    forms = "".join("v" if _vowel(stem, i) else "c" for i in range(len(stem)))
    return len(re.findall(r"v+c+", forms))


def _vowel(word, i):
    c = word[i]
    if c in VOWEL:
        return True
    return c == "y" and i > 0 and not _vowel(word, i - 1)


def _has_vowel(stem):
    return any(_vowel(stem, i) for i in range(len(stem)))


def _cvc(w):
    """Consonant, vowel, consonant at the end, where the last is not w, x or y."""
    return len(w) >= 3 and not _vowel(w, -3 + len(w)) and _vowel(w, len(w) - 2) and not _vowel(w, len(w) - 1) and w[-1] not in "wxy"


def stem(word):
    """A compact Porter-style stemmer for English. Other scripts pass through unchanged."""
    if len(word) < 3 or not word.isascii():
        return word
    w = word
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suffix in ("ing", "ed"):
            if w.endswith(suffix) and _has_vowel(w[:-len(suffix)]):
                w = w[:-len(suffix)]
                if w[-2:] in ("at", "bl", "iz"):
                    w += "e"
                elif len(w) > 1 and w[-1] == w[-2] and w[-1] not in "lsz":
                    w = w[:-1]
                break
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    for suffix, repl in (("ational", "ate"), ("ization", "ize"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("tional", "tion"), ("alism", "al"), ("ation", "ate"), ("ator", "ate"), ("iviti", "ive"), ("biliti", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous")):
        if w.endswith(suffix) and _measure(w[:-len(suffix)]) > 0:
            w = w[:-len(suffix)] + repl
            break
    for suffix in ("ement", "ment", "ance", "ence", "able", "ible", "ant", "ent", "ism", "ate", "iti", "ous", "ive", "ize", "al", "er", "ic"):
        if w.endswith(suffix) and _measure(w[:-len(suffix)]) > 1:
            w = w[:-len(suffix)]
            break
    if w.endswith("e"):
        m = _measure(w[:-1])
        if m > 1 or (m == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    return w


class Analyzer:
    def __init__(self, stemming=True, stopwords=True):
        self.stemming = stemming
        self.stopwords = STOP if stopwords else set()

    def tokens(self, text):
        """Yield (term, position, start offset, end offset). Stopwords keep their position,
        so a phrase query still respects the gaps between words."""
        pos = 0
        for raw, start, end in words(text):
            word = raw.lower().replace("'", "")
            if word not in self.stopwords:
                yield (stem(word) if self.stemming else word), pos, start, end
            pos += 1

    def terms(self, text):
        return [t[0] for t in self.tokens(text)]

    def config(self):
        return {"stemming": self.stemming, "stopwords": bool(self.stopwords)}
