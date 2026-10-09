from typing import TypedDict


class BlogState(TypedDict, total=False):
    topic: str      # what the article is about
    outline: str    # section-by-section plan
    draft: str      # first full version of the article
    final: str      # polished article after review
    filename: str   # where the article gets saved
