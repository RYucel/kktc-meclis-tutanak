"""High-throughput local CUDA FP16 Greek -> Turkish translator using CTranslate2 + NLLB-200-1.3B.

Runs on the local NVIDIA RTX 3060 12GB Tensor Cores (~600-900 chunks/min) alongside the cloud NIM
worker, writing content-addressed chunk translations to `data/cy_translations/cache_nllb.jsonl`.
`cy_translate.load_cache()` automatically merges `cache_*.jsonl` with `cache.jsonl` (giving LLM
entries in `cache.jsonl` precedence when both exist).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from cy_translate import (
    _PRIORITY_KW_RE,
    chunk_hash,
    has_greek,
    is_valid_turkish_translation,
    load_cache,
    split_greek_chunks,
)

_DEACC = str.maketrans("άέήίόύώΆΈΉΊΌΎΏ", "αεηιουωΑΕΗΙΟΥΩ")
_GREEK_CHAR_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")

_KATHAREVOUSA_SUBS = [
    (re.compile(r"\bέναρξιν\b", re.I), "έναρξη"),
    (re.compile(r"\bλήξιν\b", re.I), "λήξη"),
    (re.compile(r"\bΗμερησίας\s+Διατάξεως\b", re.I), "Ημερήσιας Διάταξης"),
    (re.compile(r"\bΒουλευτικής\s+περιόδου\b", re.I), "Κοινοβουλευτικής Περιόδου"),
    (re.compile(r"\bδιαπιστώση\s+εάν\s+υπάρχη\s+απαρτία\b", re.I), "επιβεβαιώσει αν υπάρχει απαρτία στη συνεδρίαση"),
    (re.compile(r"\bΥπάρχει\s+απαρτία\b", re.I), "Υπάρχει απαρτία στη συνεδρίαση"),
]

_TR_TERMINOLOGY_SUBS = [
    (re.compile(r"\bYunan Kıbrıslı(lar[a-zçğıöşü']*|\b)", re.I), r"Kıbrıslı Rum\1"),
    (re.compile(r"\bTürk Kıbrıslıların\b", re.I), "Kıbrıslı Türklerin"),
    (re.compile(r"\bTürk Kıbrıslılara\b", re.I), "Kıbrıslı Türklere"),
    (re.compile(r"\bTürk Kıbrıslıları\b", re.I), "Kıbrıslı Türkleri"),
    (re.compile(r"\bTürk Kıbrıslılarla\b", re.I), "Kıbrıslı Türklerle"),
    (re.compile(r"\bTürk Kıbrıslılardan\b", re.I), "Kıbrıslı Türklerden"),
    (re.compile(r"\bTürk Kıbrıslılar\b", re.I), "Kıbrıslı Türkler"),
    (re.compile(r"\bTürk Kıbrıslı\b", re.I), "Kıbrıslı Türk"),
    (re.compile(r"\bKıbrıslı Yunancılığ([a-zçğıöşü']*)", re.I), r"Kıbrıs Helenizmin\1"),
    (re.compile(r"\biki bölgelik\b", re.I), "iki bölgeli"),
    (re.compile(r"\biki topluluklu\b", re.I), "iki toplumlu"),
    (re.compile(r"\bTemsilciler Evi\b", re.I), "Temsilciler Meclisi"),
    (re.compile(r"\bMilletvekilleri Meclisi\b", re.I), "Temsilciler Meclisi"),
]


def normalize_greek_for_nllb(text: str) -> str:
    """Clean 1977 FineReader polytonic/double-accent OCR & archaic procedural forms into clean Modern Greek."""
    s = text.replace("!Ξ", "ξ").replace("c:", "ς").replace("ί:", "ς")
    s = re.sub(r"(?<=[\u0370-\u03ff])r(?=[\u0370-\u03ff])", "τ", s)
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if ch in ("\u0313", "\u0314", "\u0345"):
            continue
        if ch in ("\u0300", "\u0342"):
            out.append("\u0301")
        else:
            out.append(ch)
    s = unicodedata.normalize("NFC", "".join(out))

    def _fix_word(m: re.Match) -> str:
        w = m.group(0)
        acc = re.findall(r"[άέήίόύώΆΈΉΊΌΎΏΐΰ]", w)
        if len(acc) >= 2 and w[0] in "άέήίόύώΆΈΉΊΌΎΏ":
            return w[0].translate(_DEACC) + w[1:]
        if w in ("τήν", "τών", "τής", "καί", "νά", "μέ", "έν", "είς"):
            return w.translate(_DEACC)
        return w

    s = re.sub(r"[\u0370-\u03ff]+", _fix_word, s)
    for pat, repl in _KATHAREVOUSA_SUBS:
        s = pat.sub(repl, s)
    return s


def polish_turkish_output(text: str) -> str:
    s = text
    for pat, repl in _TR_TERMINOLOGY_SUBS:
        s = pat.sub(repl, s)
    # Strip any isolated leftover Greek chars copied from corrupted OCR tokens
    s = _GREEK_CHAR_RE.sub("", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[.;!?;])\s+")


def split_chunk_to_sentences(chunk: str, max_sent_chars: int = 460) -> list[tuple[int, str]]:
    """Split a chunk into `(para_idx, sentence)` pieces <= `max_sent_chars` so NLLB never truncates."""
    paras = [p.strip() for p in re.split(r"\n{2,}", chunk.strip()) if p.strip()]
    out: list[tuple[int, str]] = []
    for p_i, para in enumerate(paras):
        para_flat = re.sub(r"\s*\n\s*", " ", para).strip()
        raw_sents = [s.strip() for s in _SENT_SPLIT_RE.split(para_flat) if s.strip()]
        for sent in raw_sents:
            if len(sent) <= max_sent_chars:
                out.append((p_i, sent))
            else:
                # Split overlong clause-heavy sentences on commas/semicolons/spaces
                words = sent.split()
                cur = ""
                for w in words:
                    cand = f"{cur} {w}" if cur else w
                    if len(cand) <= max_sent_chars:
                        cur = cand
                    else:
                        if cur:
                            out.append((p_i, cur))
                        cur = w[:max_sent_chars]
                if cur:
                    out.append((p_i, cur))
    return out


def reassemble_paragraphs(pieces: list[tuple[int, str]]) -> str:
    if not pieces:
        return ""
    by_p: dict[int, list[str]] = {}
    for p_i, txt in pieces:
        if txt:
            by_p.setdefault(p_i, []).append(txt)
    return "\n\n".join(" ".join(sents) for _k, sents in sorted(by_p.items())).strip()


def ensure_cuda_dlls() -> None:
    llama_dir = Path(".tmp/llama")
    if llama_dir.exists():
        resolved = str(llama_dir.resolve())
        os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(resolved)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeches", default="data/cy_modern_build/speeches.jsonl")
    ap.add_argument("--cache", default="data/cy_translations/cache.jsonl")
    ap.add_argument("--write-cache", default="data/cy_translations/cache_nllb.jsonl")
    ap.add_argument("--model-dir", default=".tmp/models/nllb-1.3B-fp16")
    ap.add_argument("--priority-first", action="store_true")
    ap.add_argument("--skip-first", type=int, default=800, help="Leave top N priority chunks to NIM LLM first, then finish them at end")
    ap.add_argument("--macro-chunks", type=int, default=250, help="Chunks translated per GPU batch flush")
    ap.add_argument("--max-tokens-batch", type=int, default=6144)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--limit-chunks", type=int, default=0)
    args = ap.parse_args()

    ensure_cuda_dlls()
    import ctranslate2
    import sentencepiece as spm

    cache_path = Path(args.cache)
    write_path = Path(args.write_cache)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)

    raw_speeches = [json.loads(ln) for ln in Path(args.speeches).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.priority_first:
        def _prio_key(rec: dict) -> int:
            role = rec.get("speaker_role") or ""
            txt = rec.get("text") or ""
            is_kw = bool(_PRIORITY_KW_RE.search(txt))
            if is_kw and role == "milletvekili_bakan":
                return 0
            if is_kw and role != "metin":
                return 1
            if role == "milletvekili_bakan":
                return 2
            if is_kw:
                return 3
            if role in ("baskan", "gecici_baskan", "katip"):
                return 4
            return 5
        raw_speeches.sort(key=_prio_key)

    needed: dict[str, str] = {}
    for s in raw_speeches:
        txt = s.get("text") or ""
        if not has_greek(txt):
            continue
        for ch in split_greek_chunks(txt):
            if not has_greek(ch):
                continue
            h = chunk_hash(ch)
            if h not in cache and h not in needed:
                needed[h] = ch

    items = list(needed.items())
    if args.skip_first and len(items) > args.skip_first:
        # Process [skip_first:] first so we don't duplicate NIM's active head, then wrap around to [:skip_first]
        items = items[args.skip_first :] + items[: args.skip_first]
    if args.limit_chunks:
        items = items[: args.limit_chunks]

    print(
        f"[NLLB-CUDA] Cached={len(cache):,}, todo={len(items):,} unique chunks "
        f"(model={args.model_dir}, max_tokens_batch={args.max_tokens_batch}, beam={args.beam_size})",
        flush=True,
    )
    if not items:
        return

    sp_proc = spm.SentencePieceProcessor(str(Path(args.model_dir) / "sentencepiece.bpe.model"))
    translator = ctranslate2.Translator(
        str(args.model_dir),
        device="cuda",
        compute_type="float16",
        inter_threads=1,
    )

    t0 = time.perf_counter()
    done_chunks = 0
    skipped_live = 0

    for start in range(0, len(items), args.macro_chunks):
        # Refresh primary NIM cache periodically so we skip anything NIM translated meanwhile
        if start > 0 and (start // args.macro_chunks) % 4 == 0:
            cache = load_cache(cache_path)

        macro = [(h, ch) for h, ch in items[start : start + args.macro_chunks] if h not in cache]
        skipped_live += (len(items[start : start + args.macro_chunks]) - len(macro))
        if not macro:
            continue

        flat_toks: list[list[str]] = []
        flat_meta: list[tuple[int, int, bool, str]] = []  # (chunk_idx_in_macro, para_idx, is_greek, raw_sent)

        for c_i, (_h, ch) in enumerate(macro):
            pieces = split_chunk_to_sentences(ch)
            for p_i, sent in pieces:
                if has_greek(sent) or _GREEK_CHAR_RE.search(sent):
                    cleaned = normalize_greek_for_nllb(sent)
                    toks = ["ell_Grek"] + sp_proc.encode_as_pieces(cleaned) + ["</s>"]
                    flat_meta.append((c_i, p_i, True, str(len(flat_toks))))
                    flat_toks.append(toks)
                else:
                    flat_meta.append((c_i, p_i, False, sent))

        translated_sents: list[str] = []
        if flat_toks:
            res = translator.translate_batch(
                flat_toks,
                target_prefix=[["tur_Latn"]] * len(flat_toks),
                beam_size=args.beam_size,
                max_batch_size=args.max_tokens_batch,
                batch_type="tokens",
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                max_decoding_length=320,
            )
            translated_sents = [
                polish_turkish_output(sp_proc.decode_pieces(r.hypotheses[0][1:]))
                for r in res
            ]

        per_chunk_pieces: list[list[tuple[int, str]]] = [[] for _ in range(len(macro))]
        for c_i, p_i, is_el, ref in flat_meta:
            txt_out = translated_sents[int(ref)] if is_el else ref
            per_chunk_pieces[c_i].append((p_i, txt_out))

        written_now = 0
        with write_path.open("a", encoding="utf-8") as fh:
            for c_i, (h, src_ch) in enumerate(macro):
                tr_full = reassemble_paragraphs(per_chunk_pieces[c_i])
                if tr_full and is_valid_turkish_translation(src_ch, tr_full):
                    cache[h] = tr_full
                    fh.write(json.dumps({"h": h, "tr": tr_full}, ensure_ascii=False) + "\n")
                    written_now += 1
        done_chunks += written_now
        dt = max(time.perf_counter() - t0, 0.1)
        rate = done_chunks / dt
        rem = max(0, len(items) - (start + len(items[start : start + args.macro_chunks])))
        eta_m = (rem / max(rate, 0.01)) / 60.0
        print(
            f"  [NLLB {done_chunks + skipped_live:,}/{len(items):,}] +{written_now} cached "
            f"({rate * 60:,.0f} ch/min, elapsed {dt:.0f}s, eta {eta_m:.1f}m)",
            flush=True,
        )

    print(
        f"[NLLB-CUDA] COMPLETE: +{done_chunks:,} chunks written (skipped {skipped_live:,} already cached by NIM) "
        f"in {(time.perf_counter() - t0)/60:.1f}m!",
        flush=True,
    )


if __name__ == "__main__":
    main()
