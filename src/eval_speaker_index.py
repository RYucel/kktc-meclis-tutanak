"""Offline preview of the 3rd-pass speaker resolution using only a speakers_index.json.

No Parquet / raw archive needed: each index entry's per-year timeline ("y") is treated as
(speaker_name, year) groups, which is exactly the granularity ``recanonicalize.py`` resolves at.
Use it to review what a resolver change would do before touching the dataset:

    python src/eval_speaker_index.py space_static/speakers_index.json --show 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_resolution import SpeakerResolver, load_rosters  # noqa: E402


def load_groups(index_path: str):
    speakers = json.loads(Path(index_path).read_text(encoding="utf-8"))["speakers"]
    stats, groups = {}, []
    for x in speakers:
        yc = Counter({int(y): c for y, c in x.get("y", [])})
        if not yc and x.get("a") and x["a"][:4] == (x.get("b") or "")[:4]:
            yc[int(x["a"][:4])] = x["s"]
        stats[x["n"]] = {"count": x["s"], "years": yc, "docs": x.get("d", 0)}
        groups.extend((x["n"], y, c) for y, c in yc.items())
        rest = x["s"] - sum(yc.values())
        if rest > 0:
            groups.append((x["n"], None, rest))
    return speakers, stats, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("index", nargs="?", default="space_static/speakers_index.json")
    ap.add_argument("--rosters", default=None)
    ap.add_argument("--show", type=int, default=30, help="How many top mappings to print")
    ap.add_argument("--json", default=None, help="Optional path to dump all mappings")
    args = ap.parse_args()

    speakers, stats, groups = load_groups(args.index)
    resolver = SpeakerResolver(stats, load_rosters(args.rosters))
    by_method, final, maps = Counter(), Counter(), defaultdict(int)
    for name, year, cnt in groups:
        res = resolver.resolve(name, None, year)
        target = res.name if res else name
        final[target] += cnt
        if res and res.name != name:
            by_method[res.method] += cnt
            maps[(name, res.name, res.method)] += cnt

    verified = resolver.verified_names()
    total = sum(x["s"] for x in speakers)
    print(f"anchors: {len(resolver.anchors):,} | anchor merges: {len(resolver.report.get('anchor_merges', [])):,}")
    print(f"distinct speaker names: {len(speakers):,} -> {len(final):,}")
    print(f"names with <=2 turns:   {sum(1 for x in speakers if x['s'] <= 2):,} -> {sum(1 for c in final.values() if c <= 2):,}")
    print(f"turns re-attributed:    {sum(by_method.values()):,} of {total:,} {dict(by_method)}")
    print(
        "verified turn share:    "
        f"{sum(x['s'] for x in speakers if x['n'] in verified) / total:.2%} -> "
        f"{sum(c for k, c in final.items() if k in verified) / total:.2%}"
    )
    print("\nanchor merges:")
    for weak, strong, reason in resolver.report.get("anchor_merges", []):
        print(f"  {weak!r:40} -> {strong!r:32} [{reason}] {stats.get(weak, {}).get('count', 0):,}")
    print(f"\ntop {args.show} label mappings:")
    for (a, b, m), c in sorted(maps.items(), key=lambda kv: -kv[1])[: args.show]:
        print(f"  {c:>6,}  {a!r} -> {b!r} [{m}]")
    if args.json:
        Path(args.json).write_text(
            json.dumps([{"from": a, "to": b, "method": m, "turns": c} for (a, b, m), c in sorted(maps.items(), key=lambda kv: -kv[1])], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
