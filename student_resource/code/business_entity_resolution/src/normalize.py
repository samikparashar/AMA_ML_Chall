"""Unicode-safe normalization and open-set address helpers."""

import re
import unicodedata

import pandas as pd

LEGAL_SUFFIXES = (
    ("private limited", "pvt ltd"), ("private ltd", "pvt ltd"),
    ("public limited", "plc"), ("incorporated", "inc"),
    ("corporation", "corp"), ("company", "co"), ("limited", "ltd"),
    ("private", "pvt"), ("inc", "inc"), ("corp", "corp"),
    ("co", "co"), ("ltd", "ltd"), ("llc", "llc"), ("llp", "llp"),
    ("plc", "plc"),
)
ADDRESS_ABBREVIATIONS = (
    ("boulevard", "blvd"), ("highway", "hwy"), ("avenue", "ave"),
    ("street", "st"), ("road", "rd"), ("drive", "dr"), ("lane", "ln"),
    ("parkway", "pkwy"), ("place", "pl"), ("square", "sq"),
    ("terrace", "ter"), ("court", "ct"), ("apartment", "apt"),
    ("building", "bldg"),
)
# \w (Python re, Unicode mode) matches letters/digits but not combining marks, so
# Indic dependent vowel signs (matras), virama, and anusvara/candrabindu -- essential,
# not decorative, in Devanagari/Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/
# Malayalam -- would otherwise be silently stripped here. Explicitly keep that whole
# Unicode range so transliterate.py downstream still has them to work with.
_NON_WORD = re.compile(r"[^\w\s,ऀ-ൿ]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _base(text: object, keep_commas: bool = False) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value).lower().replace("&", " and ")
    value = _NON_WORD.sub("", value)
    value = _SPACE.sub(" ", value).strip(" ,")
    if keep_commas:
        value = re.sub(r"\s*,\s*", ",", value)
    else:
        value = value.replace(",", " ")
    return value


def _replace_phrases(value: str, replacements: tuple) -> str:
    for old, new in replacements:
        value = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, value)
    return _SPACE.sub(" ", value).strip()


def normalize_name(text: object) -> str:
    return _replace_phrases(_base(text), LEGAL_SUFFIXES)


def normalize_address(text: object) -> str:
    return _replace_phrases(_base(text, keep_commas=True), ADDRESS_ABBREVIATIONS)


def extract_city(address_norm: object) -> str:
    """Return the second-to-last comma segment; empty when structure is unclear."""
    value = "" if address_norm is None else str(address_norm)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts[-2] if len(parts) >= 3 else ""


def phonetic_key(name_norm: object) -> str:
    value = "" if name_norm is None else str(name_norm).strip()
    if not value:
        return ""
    if all(ord(char) < 128 for char in value):
        try:
            import jellyfish
            return jellyfish.metaphone(value)
        except ImportError:
            pass
    result = []
    vowels = set("aeiou")
    for token in re.findall(r"\w+", value, flags=re.UNICODE):
        chars = [char for index, char in enumerate(token) if index == 0 or char not in vowels]
        for char in chars:
            if not result or result[-1] != char:
                result.append(char)
    return "".join(result)


def add_normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["business_name_norm"] = result["business_name"].map(normalize_name)
    result["business_address_norm"] = result["business_address"].map(normalize_address)
    result["city_token"] = result["business_address_norm"].map(extract_city)
    result["phonetic_key"] = result["business_name_norm"].map(phonetic_key)
    return result
