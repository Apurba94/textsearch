"""TextSearch: a full-text search engine."""
from .analyze import Analyzer, stem
from .index import Index, decode_postings, encode_postings
from .search import QueryError, Searcher, edit_distance

__all__ = ["Analyzer", "stem", "Index", "Searcher", "QueryError", "edit_distance", "encode_postings", "decode_postings"]
