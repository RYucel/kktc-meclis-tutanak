"""Turn cy_parse.py output (one JSONL per session volume) into KKTC dataset rows (documents + speeches).

One document per sitting. doc_id: 'Kıbrıs Cumhuriyeti Temsilciler Meclisi/<Synodos> <date>'.
source_url points at the official parliament.cy volume PDF with '#page=N' so citations open the right page.

Usage:
  python src/cy_build.py --vol A turns_A.jsonl --vol B turns_B.jsonl ... --out data/cy_build
Writes <out>/documents.jsonl, <out>/speeches.jsonl and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FOLDER = "Kıbrıs Cumhuriyeti Temsilciler Meclisi"
VOLUMES = {  # synodos letter -> (yasama yılı, label, official PDF)
    "A": (1, "Α΄ Σύνοδος 1960-61", "https://www.parliament.cy/images/media/redirectfile/A Vouleftiki periodos -Synodos A - 16.08.1960-07.081961.pdf"),
    "B": (2, "Β΄ Σύνοδος 1961-62", "https://www.parliament.cy/images/media/redirectfile/Α΄ Βουλευτική Περίοδος - Σύνοδος Β΄ - 16.08.1961-10.07.1962.pdf"),
    "C": (3, "Γ΄ Σύνοδος 1962-63", "https://www.parliament.cy/images/media/redirectfile/Synodos-C.pdf"),
    "D": (4, "Δ΄ Σύνοδος 1963-64", "https://www.parliament.cy/images/media/redirectfile/Synodos-D.pdf"),
}
TEXT_SOURCE = "pdf_tesseract_ell_tur"
LANG_LABEL = {"el": "Yunanca", "tr": "Türkçe", "mixed": "Yunanca/Türkçe"}


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def build(vol_files: list[tuple[str, Path]]):
    docs, speeches = [], []
    for vol, path in vol_files:
        yy, label, pdf = VOLUMES[vol]
        turns = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        by_sitting: dict[int, list[dict]] = defaultdict(list)
        for t in turns:
            by_sitting[t["sitting"]].append(t)
        for birlesim, (sit, ts) in enumerate(sorted(by_sitting.items()), start=1):
            tarih = next((t["tarih"] for t in ts if t.get("tarih")), None)
            doc_id = f"{FOLDER}/Synodos {vol} - {birlesim:02d}. birleşim{(' ' + tarih) if tarih else ''}"
            dslug = slug(f"KC_{vol}_{birlesim:02d}_{tarih or ''}")
            first_page = min((t["page"] for t in ts if t.get("page")), default=None)
            for i, t in enumerate(ts):
                page = t.get("page") or first_page
                speeches.append({
                    "doc_id": doc_id, "meclis": "KC", "donem": None, "yasama_yili": yy, "birlesim": birlesim, "oturum": 1,
                    "tarih": tarih, "speaker_raw": t["speaker_raw"], "speaker_name": t["speaker_name"],
                    "speaker_role": t["speaker_role"], "speaker_title": None, "constituency": None, "modifiers": LANG_LABEL.get(t.get("lang") or ""),
                    "text_source": TEXT_SOURCE, "source_file": f"{label}.pdf", "source_url": f"{pdf}#page={page}" if page else pdf,
                    "text": t["text"], "char_count": len(t["text"]), "word_count": len(t["text"].split()), "turn_index": i,
                    "speech_id": f"{dslug}#o1#t{i:04d}", "speaker_name_parsed": t["speaker_raw"], "canon_method": t["canon_method"],
                })
            text = "\n\n".join(f"{t['speaker_raw']}:\n{t['text']}" for t in ts)
            people = sorted({t["speaker_name"] for t in ts if t["speaker_role"] in ("milletvekili_bakan",)})
            chairs = sorted({t["speaker_name"] for t in ts if t["speaker_role"] == "baskan"})
            docs.append({
                "doc_id": doc_id, "meclis": "KC", "donem": None, "yasama_yili": yy, "birlesim": birlesim, "tarih": tarih,
                "oturum_sayisi": 1, "konusma_sayisi": len(ts), "konusmacilar": people, "baskanlar": chairs, "text": text,
                "char_count": len(text), "page_count": len({t.get("page") for t in ts if t.get("page")}),
                "text_source": TEXT_SOURCE, "source_file": f"{label}.pdf",
                "source_url": f"{pdf}#page={first_page}" if first_page else pdf, "sha256": None,
            })
    return docs, speeches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", nargs=2, action="append", metavar=("LETTER", "JSONL"), required=True)
    ap.add_argument("--out", default="data/cy_build")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs, speeches = build([(v, Path(p)) for v, p in a.vol])
    for name, rows in (("documents", docs), ("speeches", speeches)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "documents": len(docs), "speeches": len(speeches),
        "dated_documents": sum(1 for d in docs if d["tarih"] and len(d["tarih"]) == 10),
        "lang": dict(Counter(s["modifiers"] for s in speeches)),
        "turkish_turns_by_speaker": Counter(s["speaker_name"] for s in speeches if s["modifiers"] == "Türkçe").most_common(25),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
