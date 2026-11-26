"""Conservative text normalization with an offset map back to the original text.

Rule (research_foundation §9.2): never strip or rewrite numbers, units, negation cues, comparatives or
symbols such as "≥", "<", "%". Therefore NO Unicode compatibility normalization (NFKC): it rewrites
superscripts ("10⁹/L" → "109/L") and other characters that carry clinical meaning.

What is changed (only):
* invisible characters removed: zero-width space/joiners, BOM, soft hyphen;
* exotic space characters (no-break, thin, narrow no-break, …) → plain space;
* typographic ligatures (ﬁ, ﬂ, …) expanded;
* runs of spaces/tabs collapsed to one space; spaces at line ends removed;
* runs of ≥ 3 newlines collapsed to a blank line ("\\n\\n"); "\\r\\n" / "\\r" → "\\n".

``normalize`` returns the new text and ``src``: for every output character, the index of the input
character it came from, plus one final entry = len(input). Use ``map_span`` to move a span from the
original text into the normalized text (e.g. a gold answer span or an entity mention).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", "­"}
_SPACES = {" ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
           " ", " ", " ", " ", "　", "\t", "\x0b", "\x0c"}
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


@dataclass(frozen=True)
class Normalized:
    text: str
    # len(text) + 1 entries; src[i] = input index of output char i; src[-1] = len(input).
    # None when normalization changed nothing (identity map; avoids a large list for clean texts).
    src: list[int] | None

    @property
    def changed(self) -> bool:
        return self.src is not None

    def map_span(self, start: int, end: int) -> tuple[int, int]:
        """Map an input span [start, end) to the output span covering the same surviving characters."""
        if self.src is None:
            return start, end
        s = bisect.bisect_left(self.src, start, 0, len(self.src) - 1)
        e = bisect.bisect_left(self.src, end, 0, len(self.src) - 1)
        return s, max(s, e)


_NEEDS_WORK = re.compile("[" + "".join(sorted(_INVISIBLE | _SPACES | set(_LIGATURES))) + r"\r]| {2,}| \n|\n |\n{3,}")


def normalize(text: str) -> Normalized:
    if not _NEEDS_WORK.search(text) and not text.startswith((" ", "\n")):
        return Normalized(text, None)
    out: list[str] = []
    src: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _INVISIBLE:
            i += 1
            continue
        if ch == "\r":
            ch = "\n"
            if i + 1 < n and text[i + 1] == "\n":
                i += 1  # "\r\n" → one "\n" (mapped to the "\n")
        if ch in _SPACES:
            ch = " "
        if ch in _LIGATURES:
            for c in _LIGATURES[ch]:
                out.append(c); src.append(i)
            i += 1
            continue
        if ch == " ":
            if out and out[-1] in (" ", "\n"):
                i += 1
                continue  # collapse runs; no space at line start
        elif ch == "\n":
            while out and out[-1] == " ":  # no trailing space before a newline
                out.pop(); src.pop()
            if len(out) >= 2 and out[-1] == "\n" and out[-2] == "\n":
                i += 1
                continue  # at most one blank line
        out.append(ch); src.append(i)
        i += 1
    src.append(n)
    return Normalized("".join(out), src)
