"""Cleaning and filtering rules shared by data_download.py and data_prep.py."""
import re

EOT = "<|endoftext|>"   # document separator (special token in the tokenizer)
MIN_CHARS = 500         # drop short stub articles / pages

# lines that start with a year / date, e.g. "1881 - ...", "* MÖ 330", "44 BC"
LIST_LINE = re.compile(r"^\s*[-*•]?\s*(\d{1,4}|M[ÖS]\s*\d+|\d+\s*(BC|AD))\b")
LIST_MIN_LINES = 10
LIST_RATIO = 0.4

# leftovers of templates removed from the Wikipedia dump: "Ankara ( ; , )" or "(; d. 1881)"
EMPTY_PARENS = re.compile(r"\s*\([\s,;:–\-]*\)")
LEADING_PUNCT = re.compile(r"\(\s*[,;][,;\s]*")

# section headings after which a Wikipedia article is only references / link lists
REFERENCE_HEADINGS = {"Kaynakça", "Kaynaklar", "Notlar", "Dış bağlantılar", "Ayrıca bakınız",
                      "References", "Further reading", "External links", "See also"}

def strip_reference_sections(text):
    """Cut the article at the first line that is exactly a reference-section heading."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in REFERENCE_HEADINGS:
            return "\n".join(lines[:i])
    return text

def strip_category_tail(text):
    """Remove category lines (short, no final punctuation) at the end of a document."""
    lines = text.rstrip().splitlines()
    end = len(lines)
    while end > 0:
        s = lines[end - 1].strip()
        if s and len(s) < 80 and not s.endswith((".", "!", "?", ":", '"', ")")):
            end -= 1
        else:
            break
    if len(lines) - end >= 2:
        lines = lines[:end]
    return "\n".join(lines)

# whitespace left in the Wikipedia dump, measured 4 Nov per 1M characters: 1303 spaces at line start
# (removed list bullets), 676 at line end, 348 double spaces, 208 non-breaking spaces; web text 0-13
SPACES = re.compile(r"[ \t  -   　]+")   # space, tab, nbsp, thin space, ...
ZERO_WIDTH = re.compile(r"[​-‍﻿]")                    # invisible: zero-width space, joiners, BOM
BLANK_LINES = re.compile(r"\n{3,}")

def normalize_whitespace(text):
    """One plain space between words, none at line start or end, at most one empty line in a row."""
    text = ZERO_WIDTH.sub("", text)
    lines = (SPACES.sub(" ", line).strip(" ") for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", "\n".join(lines))

def clean(text):
    text = text.replace(EOT, "")             # the separator must only ever appear between documents
    text = EMPTY_PARENS.sub("", text)        # "( ; , )"      -> ""
    text = LEADING_PUNCT.sub("(", text)      # "(; d. 1881)"  -> "(d. 1881)"
    text = strip_reference_sections(text)    # "Kaynakça", "See also", ... -> cut
    text = strip_category_tail(text)         # short category lines at the end -> cut
    return normalize_whitespace(text).strip()   # last: the cuts above can leave spaces behind

def is_list_page(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < LIST_MIN_LINES:
        return False
    hits = sum(1 for l in lines if LIST_LINE.match(l))
    return hits / len(lines) > LIST_RATIO

# Repetition thresholds, from the Gopher paper (Rae et al. 2021, table A1)
DUP_LINE_FRAC = 0.30    # share of lines that are copies of an earlier line
DUP_CHAR_FRAC = 0.20    # share of characters that sit in those copied lines

def is_repetitive(text):
    """True if the document repeats itself: menus, "read more" blocks, spam, copy-pasted lists.
    A line counts as a duplicate when the same line already appeared earlier in the document."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    seen, dup_lines, dup_chars = set(), 0, 0
    for line in lines:
        if line in seen:
            dup_lines += 1
            dup_chars += len(line)
        else:
            seen.add(line)
    total_chars = sum(len(l) for l in lines)
    return dup_lines / len(lines) > DUP_LINE_FRAC or dup_chars / total_chars > DUP_CHAR_FRAC

def reject_reason(text):
    """None if the (already cleaned) document is kept, otherwise a short reason for the log."""
    if len(text) < MIN_CHARS:
        return "short"
    if is_list_page(text):
        return "list"
    if is_repetitive(text):
        return "repetitive"
    return None
