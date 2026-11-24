"""OCR quality scoring for KKTC Meclis minutes.

Builds a Turkish vocabulary from the born-digital transcripts (doc/docx/rtf) and scores every
document (and optionally every speech turn) for OCR damage:

  oov       share of alphabetic words (len >= 4) that never occur in the clean vocabulary
  mixed     share of tokens that glue letters with digits/symbols ("K^TC", "4din", "KK1lC*nin")
  frag      share of 1-2 letter tokens that are not real Turkish short words ("mi 1 letveki 1 leri")
  score     0..1 combined damage score (higher = worse)

Usage:
  python src/ocr_quality.py --data-dir data/hf_dataset/data --out data/ocr_quality
Outputs:
  <out>/vocab.parquet            word, n   (clean-source vocabulary)
  <out>/doc_quality.parquet      doc_id, meclis, donem, tarih, text_source, source_file, words, oov, mixed, frag, score
  <out>/spaced_docs.txt         letter-spaced PDF documents (input for reprocess_docs.py)
  <out>/noisy_docs.txt          OCR-noisy PDF documents, worst first (input for reocr_vlm.py)
  <out>/summary.json             distribution by meclis / decade / text_source and the worst documents
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

CLEAN_SOURCES = ("doc", "docx", "rtf", "pdf_digital")
WORD_RE = re.compile(r"[a-zçğıöşüâîû]+")
TOKEN_RE = re.compile(r"\S+")
# letters glued to digits or OCR junk symbols inside one token
MIXED_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû][0-9^*|\\{}<>~_=@#$%]|[0-9^*|\\{}<>~_=@#$%][A-Za-zÇĞİÖŞÜçğıöşüâîû]")
SHORT_OK = {
    "o", "a", "e", "ve", "de", "da", "ki", "mi", "mı", "mu", "mü", "bu", "şu", "ne", "ya", "en", "ise", "ile",
    "bir", "ben", "sen", "biz", "siz", "ama", "her", "hiç", "çok", "az", "on", "iki", "üç", "dört", "beş",
    "an", "ad", "al", "at", "ay", "ek", "el", "er", "eş", "ev", "iç", "il", "iş", "ok", "ol", "ön", "su", "üs", "üst",
    "bey", "dr", "no", "vs", "tc", "ab", "bm", "ab", "km", "tl", "yıl", "gün", "saat",
}


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _p(path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def build_vocab(con, docs: str, out: Path, min_count: int = 3) -> set[str]:
    vp = out / "vocab.parquet"
    out.mkdir(parents=True, exist_ok=True)
    if not vp.exists():
        srcs = ",".join(f"'{s}'" for s in CLEAN_SOURCES)
        con.execute(
            f"""
            COPY (
              SELECT w AS word, COUNT(*) AS n FROM (
                SELECT unnest(regexp_extract_all(lower(replace(replace(text,'İ','i'),'I','ı')), '[a-zçğıöşüâîû]+')) AS w
                FROM read_parquet('{_p(docs)}') WHERE text_source IN ({srcs})
              ) GROUP BY w HAVING COUNT(*) >= {min_count}
            ) TO '{_p(vp)}' (FORMAT PARQUET)
            """
        )
    return set(pq.read_table(vp, columns=["word"]).column(0).to_pylist())


def score_text(text: str, vocab: set[str]) -> dict:
    toks = TOKEN_RE.findall(text or "")
    n_tok = len(toks) or 1
    mixed = sum(1 for t in toks if MIXED_RE.search(t))
    words = WORD_RE.findall(tr_lower(text or ""))
    long_w = [w for w in words if len(w) >= 4]
    oov = sum(1 for w in long_w if w not in vocab)
    short = [w for w in words if len(w) <= 2]
    frag = sum(1 for w in short if w not in SHORT_OK)
    r_oov = oov / (len(long_w) or 1)
    r_mixed = mixed / n_tok
    r_frag = frag / (len(words) or 1)
    score = min(1.0, 0.6 * r_oov + 2.0 * r_mixed + 1.0 * r_frag)
    return {"words": len(words), "oov": round(r_oov, 4), "mixed": round(r_mixed, 4), "frag": round(r_frag, 4), "score": round(score, 4)}


def _norm(w: str) -> str:
    """Same normalisation the Space applies to queries: Turkish lowercase, dotless i folded to i."""
    return tr_lower(w).replace("ı", "i")


def _deletes(w: str) -> set[str]:
    return {w[:i] + w[i + 1:] for i in range(len(w))}


def build_ocr_variants(con, docs: Path, out: Path, min_word_n: int = 20, min_var_n: int = 2, max_per_word: int = 8) -> dict:
    """Map frequent clean words to OCR-damaged spellings seen in scanned documents (edit distance 1, unique target).

    The Space ORs these variants into a search term, so 'karşı' also finds 'karjı' / 'karsı' / 'kar şı'-free
    one-off misreads. Keys and values use the query normalisation (_norm).
    """
    vocab = {}
    for w, n in con.execute(
        f"SELECT word, n FROM read_parquet('{_p(out / 'vocab.parquet')}')"
    ).fetchall():
        k = _norm(w)
        vocab[k] = vocab.get(k, 0) + n
    frequent = {w: n for w, n in vocab.items() if n >= min_word_n and len(w) >= 5}
    del_index: dict[str, set[str]] = {}
    for w in frequent:
        for d in _deletes(w) | {w}:
            del_index.setdefault(d, set()).add(w)
    oov = con.execute(
        f"""
        SELECT w, COUNT(*) n FROM (
          SELECT unnest(regexp_extract_all(replace(lower(replace(replace(text,'İ','i'),'I','ı')),'ı','i'), '[a-zçğıöşüâîû]+')) AS w
          FROM read_parquet('{_p(docs)}') WHERE text_source LIKE 'pdf%'
        ) WHERE length(w) >= 4 GROUP BY w HAVING COUNT(*) >= {min_var_n}
        """
    ).fetchall()
    variants: dict[str, list[tuple[str, int]]] = {}
    for tok, n in oov:
        if tok in vocab:
            continue
        cands = set(del_index.get(tok, ()))  # tok is a deletion of w (missing letter)
        for d in _deletes(tok):
            cands |= del_index.get(d, set())  # substitution (shared deletion) or extra letter (d == w)
        cands = {c for c in cands if abs(len(c) - len(tok)) <= 1}
        if len(cands) == 1:
            variants.setdefault(cands.pop(), []).append((tok, n))
    result = {
        w: [t for t, _ in sorted(vs, key=lambda x: -x[1])[:max_per_word]]
        for w, vs in variants.items()
        if sum(n for _, n in vs) >= 3
    }
    (out / "ocr_variants.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/hf_dataset/data")
    ap.add_argument("--out", default="data/ocr_quality")
    ap.add_argument("--memory-limit", default="1.5GB")
    ap.add_argument("--worst", type=int, default=50)
    ap.add_argument("--variants", action="store_true", help="Also build ocr_variants.json for OCR-tolerant search")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = Path(args.data_dir) / "documents.parquet"
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{_p(out / '.tmp')}'")
    con.execute("SET preserve_insertion_order=false")
    vocab = build_vocab(con, docs, out)
    print(f"vocab: {len(vocab):,} words ({time.perf_counter() - t0:.0f}s)")

    rows = []
    meta_cols = ["doc_id", "meclis", "donem", "tarih", "text_source", "source_file"]
    pf = pq.ParquetFile(docs)
    for batch in pf.iter_batches(batch_size=64, columns=meta_cols + ["text"]):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            r = {c: d[c][i] for c in meta_cols}
            r.update(score_text(d["text"][i], vocab))
            rows.append(r)
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, out / "doc_quality.parquet")
    con.register("q", tbl)

    def grp(expr):
        return [
            dict(zip(["group", "docs", "words_m", "median_score", "p90_score", "bad_docs"], r))
            for r in con.execute(
                f"""SELECT {expr} g, COUNT(*), ROUND(SUM(words)/1e6,2), ROUND(MEDIAN(score),3), ROUND(QUANTILE_CONT(score,0.9),3),
                           SUM(CASE WHEN score >= 0.15 THEN 1 ELSE 0 END)
                    FROM q GROUP BY g ORDER BY MIN(tarih)"""
            ).fetchall()
        ]

    summary = {
        "vocab_words": len(vocab),
        "thresholds": {"bad": 0.15},
        "by_text_source": grp("text_source"),
        "by_meclis": grp("meclis"),
        "by_decade": grp("substr(tarih,1,3) || '0s'"),
        "worst": [
            dict(zip(["doc_id", "tarih", "text_source", "source_file", "words", "oov", "mixed", "frag", "score"], r))
            for r in con.execute(
                f"SELECT doc_id, tarih, text_source, source_file, words, oov, mixed, frag, score FROM q WHERE words > 500 ORDER BY score DESC LIMIT {args.worst}"
            ).fetchall()
        ],
    }
    spaced = [r[0] for r in con.execute(
        "SELECT doc_id FROM q WHERE text_source LIKE 'pdf%' AND frag >= 0.3 ORDER BY doc_id").fetchall()]
    (out / "spaced_docs.txt").write_text("\n".join(spaced) + "\n", encoding="utf-8")
    summary["spaced_docs"] = len(spaced)
    noisy = [r[0] for r in con.execute(
        "SELECT doc_id FROM q WHERE text_source LIKE 'pdf%' AND frag < 0.3 AND score >= 0.15 ORDER BY score DESC").fetchall()]
    (out / "noisy_docs.txt").write_text("\n".join(noisy) + "\n", encoding="utf-8")
    summary["noisy_docs"] = len(noisy)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.variants:
        v = build_ocr_variants(con, docs, out)
        print(f"ocr_variants.json: {len(v):,} words, {sum(len(x) for x in v.values()):,} variants")
    print(f"scored {len(rows):,} documents in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
