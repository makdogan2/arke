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

def clean(text):
    text = text.replace(EOT, "")             # the separator must only ever appear between documents
    text = EMPTY_PARENS.sub("", text)        # "( ; , )"      -> ""
    text = LEADING_PUNCT.sub("(", text)      # "(; d. 1881)"  -> "(d. 1881)"
    return strip_category_tail(text).strip()

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
    TODO(Mehmet): implement with the two thresholds above.
      - take the non-empty lines (strip them)
      - a line is a "duplicate" if the same line already appeared earlier in the document
      - repetitive if  duplicates / lines > DUP_LINE_FRAC
                   or  chars in duplicates / chars in all lines > DUP_CHAR_FRAC
    """
    raise NotImplementedError("TODO(Mehmet): is_repetitive")

def reject_reason(text):
    """None if the (already cleaned) document is kept, otherwise a short reason for the log."""
    if len(text) < MIN_CHARS:
        return "short"
    if is_list_page(text):
        return "list"
    if is_repetitive(text):
        return "repetitive"
    return None
