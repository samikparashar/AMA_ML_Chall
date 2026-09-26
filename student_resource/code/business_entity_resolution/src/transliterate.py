"""GPU-batched transliteration of Indic-script business text into a common
Latin representation.

Why this exists: blocking.py / blocking_faiss.py both hash raw characters
into char n-grams (HashingVectorizer(analyzer="char_wb", ...)) and build
composite/exact keys from raw normalized strings. A true match where one
side is written in Latin script and the other in Devanagari, Bengali,
Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, or Malayalam shares zero
characters with its Latin counterpart, so it gets zero n-gram overlap, zero
composite-key/exact-match hits, and zero ANN/cosine signal -- a full miss
upstream of scoring, not a ranking problem. This module Latinizes text from
those scripts so both sides land in the same character space before they
reach the vectorizers/keys. It does not touch the original
business_name_norm/business_address_norm fields -- callers add parallel
*_translit columns and feed those to vectorizers/keys instead.

Approach: transliteration data comes from `unidecode`'s per-codepoint ASCII
tables (an existing, maintained set of Unicode -> ASCII mappings that
already covers these scripts), refined with one general Indic-script rule:
re-add a bare consonant's inherent "a" unless the next character is a
dependent vowel sign or virama that supplies/cancels it. (unidecode's own
per-codepoint entries already strip that inherent vowel unconditionally,
e.g. Devanagari "न" -> "n", which is only correct when a vowel sign
follows -- "नमस्ते" needs it back to read "namaste" instead of "nmste".)
Unicode intentionally aligns these nine scripts' code points to the same
ISCII-derived relative layout, so one small set of relative-offset ranges
identifies consonants/vowel-signs/virama across all of them.

The actual per-character lookup is a single batched gather + boolean-masked
scatter on the GPU (Metal/MPS or CUDA -- whatever `_default_device()`
resolves to), matching how blocking.py already treats its coarse
vectors/cluster assignment as batched tensor ops, not a per-row Python
loop. Input packing (str -> flat codepoint array) and output unpacking use
vectorized numpy/bytes encoding rather than a per-character Python loop, so
the only per-row Python-level work is O(rows) string slicing, not
O(characters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from unidecode import unidecode

from .normalize import extract_city, phonetic_key

# Devanagari through Malayalam is one contiguous Unicode range that covers
# every script named in the brief (Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Tamil, Telugu, Kannada, Malayalam).
_INDIC_START = 0x0900
_TABLE_SIZE = 0x0D80  # covers ASCII/Latin (identity) + all of the above
_MAX_OUT = 4  # >99% of codepoints in range transliterate to <=3 chars; rarer longer ones truncate

# Unicode's Indic blocks mirror the ISCII layout, so these block-relative
# offsets identify consonants / dependent-vowel-signs / virama consistently
# across all nine scripts above.
_CONSONANT_REL = range(0x15, 0x3A)
_VOWEL_CARRIER_REL = range(0x3E, 0x4E)  # dependent vowel signs (matras) + virama
_BLOCK_STARTS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

_TABLE_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.zeros((_TABLE_SIZE, _MAX_OUT), dtype=np.int64)
    is_consonant = np.zeros(_TABLE_SIZE, dtype=bool)
    is_vowel_carrier = np.zeros(_TABLE_SIZE, dtype=bool)
    for codepoint in range(_TABLE_SIZE):
        text = chr(codepoint) if codepoint < _INDIC_START else unidecode(chr(codepoint)).lower()
        codes = [ord(char) for char in text[:_MAX_OUT]]
        table[codepoint, : len(codes)] = codes
        if codepoint >= _INDIC_START:
            for block_start in _BLOCK_STARTS:
                if block_start <= codepoint < block_start + 0x80:
                    relative = codepoint - block_start
                    is_consonant[codepoint] = relative in _CONSONANT_REL
                    is_vowel_carrier[codepoint] = relative in _VOWEL_CARRIER_REL
                    break
    return table, is_consonant, is_vowel_carrier


def _get_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = str(device)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    table, is_consonant, is_vowel_carrier = _build_table()
    tensors = (
        torch.from_numpy(table).to(device),
        torch.from_numpy(is_consonant).to(device),
        torch.from_numpy(is_vowel_carrier).to(device),
    )
    _TABLE_CACHE[key] = tensors
    return tensors


def _pack_codepoints(strings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten strings into one codepoint array plus per-row [start, stop) offsets."""
    lengths = np.fromiter((len(s) for s in strings), dtype=np.int64, count=len(strings))
    offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.zeros(int(offsets[-1]), dtype=np.uint32)
    pos = 0
    for s in strings:
        n = len(s)
        if n:
            flat[pos : pos + n] = np.frombuffer(s.encode("utf-32-le"), dtype="<u4")
        pos += n
    return flat, offsets


def transliterate_series(series: pd.Series, device: torch.device | None = None) -> pd.Series:
    """Latinize Indic-script characters in a text series via one batched GPU pass."""
    device = device or _default_device()
    strings = series.fillna("").astype(str).tolist()
    flat_cp, offsets = _pack_codepoints(strings)
    if flat_cp.size == 0:
        return series.copy()

    table, is_consonant, is_vowel_carrier = _get_table(device)

    cp_cpu = flat_cp.astype(np.int64)
    in_table = cp_cpu < _TABLE_SIZE
    safe_idx_np = np.where(in_table, cp_cpu, 0)
    cp_gpu = torch.from_numpy(cp_cpu).to(device)
    safe_idx = torch.from_numpy(safe_idx_np).to(device)
    in_table_t = torch.from_numpy(in_table).to(device)

    replaced = table[safe_idx].clone()  # [N, MAX_OUT] -- the batched GPU gather
    out_of_table = ~in_table_t
    replaced[out_of_table, 0] = cp_gpu[out_of_table]
    replaced[out_of_table, 1:] = 0

    consonant_flag = is_consonant[safe_idx] & in_table_t
    vowel_carrier_flag = is_vowel_carrier[safe_idx] & in_table_t

    row_end = np.zeros(flat_cp.size, dtype=bool)
    row_lengths = np.diff(offsets)
    row_end[offsets[1:][row_lengths > 0] - 1] = True
    row_end_t = torch.from_numpy(row_end).to(device)

    next_vowel_carrier = torch.cat(
        [vowel_carrier_flag[1:], torch.zeros(1, dtype=torch.bool, device=device)]
    )
    next_vowel_carrier = next_vowel_carrier & ~row_end_t
    inject_a = consonant_flag & ~next_vowel_carrier  # restore the dropped inherent "a"

    out_len = (replaced != 0).sum(dim=1)
    can_extend = inject_a & (out_len < _MAX_OUT)
    write_pos = torch.clamp(out_len, max=_MAX_OUT - 1)
    row_idx = torch.arange(replaced.shape[0], device=device)
    replaced[row_idx[can_extend], write_pos[can_extend]] = ord("a")
    out_len = torch.where(can_extend, out_len + 1, out_len)

    replaced_np = replaced.cpu().numpy()
    out_len_np = out_len.cpu().numpy()

    col_idx = np.arange(_MAX_OUT)
    mask = col_idx[None, :] < out_len_np[:, None]
    flat_out = replaced_np[mask].astype("<u4")
    joined = flat_out.tobytes().decode("utf-32-le") if flat_out.size else ""

    cum_out_len = np.zeros(out_len_np.size + 1, dtype=np.int64)
    np.cumsum(out_len_np, out=cum_out_len[1:])
    row_out_offsets = cum_out_len[offsets]

    result = [joined[row_out_offsets[i] : row_out_offsets[i + 1]] for i in range(len(strings))]
    return pd.Series(result, index=series.index)


def add_translit_columns(frame: pd.DataFrame, device: torch.device | None = None) -> pd.DataFrame:
    """Add Latinized copies of name/address/city/phonetic fields.

    Leaves business_name_norm/business_address_norm/city_token/phonetic_key
    untouched -- callers feed the *_translit columns to vectorizers/keys
    instead. Idempotent (skips columns already present) so the corpus-build
    path and the per-batch query-transform path can both call this freely.
    """
    need_name = "business_name_translit" not in frame.columns
    need_address = "business_address_translit" not in frame.columns
    need_city = "city_token_translit" not in frame.columns
    need_phonetic = "phonetic_key_translit" not in frame.columns
    if not (need_name or need_address or need_city or need_phonetic):
        return frame

    result = frame.copy()
    if need_name:
        result["business_name_translit"] = transliterate_series(result["business_name_norm"], device)
    if need_address:
        result["business_address_translit"] = transliterate_series(result["business_address_norm"], device)
    if "city_token_translit" not in result.columns:
        result["city_token_translit"] = result["business_address_translit"].map(extract_city)
    if "phonetic_key_translit" not in result.columns:
        result["phonetic_key_translit"] = result["business_name_translit"].map(phonetic_key)
    return result
