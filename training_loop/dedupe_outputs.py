# training_loop/dedupe_outputs.py
from __future__ import annotations
import argparse, hashlib, re, shutil
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(".").resolve()
TL = ROOT / "training_loop"

INJECTED_DIR = TL / "injected_outputs"
KB_DIR       = ROOT / "knowledge_base" / "training_modules" / "scenario_runs"
QUAR_DIR     = TL / "quarantine_dupes"  # where we move duplicates

TXT_GLOB = "scenario_*_output*.txt"

_TS_LINES = (
    re.compile(r"^\(Generated on .*\)\s*$"),
    re.compile(r"^\(Evaluated on .*\)\s*$"),
)

_NO_VARIANT = re.compile(r"(.*)_no\d{2}(\.txt|\.metrics\.json)$", re.IGNORECASE)
_SCEN_NUM   = re.compile(r"scenario_(\d{3})_output", re.IGNORECASE)

def canonicalize(text: str) -> str:
    """Strip volatile lines, normalize whitespace/newlines for loose matching."""
    lines = []
    for ln in text.splitlines():
        # drop timestampy lines
        if any(pat.match(ln.strip()) for pat in _TS_LINES):
            continue
        lines.append(ln.rstrip())  # trim trailing spaces
    canon = "\n".join(lines).strip()
    # collapse runs of >2 blank lines
    canon = re.sub(r"\n{3,}", "\n\n", canon)
    return canon

def content_key(path: Path, loose: bool) -> Tuple[str, str]:
    """
    Returns (group_key, hash) used to detect duplicates:
      - group_key: scenario number if present, else stem without _noXX
      - hash: md5 of (raw or canonicalized) text
    """
    stem = path.name
    m = _NO_VARIANT.match(stem)
    if m:
        stem = m.group(1) + path.suffix  # remove _noXX from group key

    scen = _SCEN_NUM.search(path.name)
    group = scen.group(0) if scen else stem  # e.g., "scenario_041_output"

    text = path.read_text(encoding="utf-8", errors="ignore")
    data = canonicalize(text) if loose else text
    digest = hashlib.md5(data.encode("utf-8")).hexdigest()
    return group, digest

def pick_canonical(files: List[Path]) -> Path:
    """
    Choose which file to keep within a duplicate set:
      1) Prefer file WITHOUT _noXX
      2) If tie, prefer shorter name (usually earlier runs)
      3) If tie, prefer newest mtime (latest run)
    """
    def score(p: Path):
        has_no = 1 if _NO_VARIANT.match(p.name) else 0  # 0 is better
        return (has_no, len(p.name), -p.stat().st_mtime)
    return sorted(files, key=score)[0]

def dedupe_folder(folder: Path, *, loose: bool, apply: bool) -> Tuple[int, int]:
    """
    De-dupe within a single folder. Returns (kept_count, quarantined_count).
    """
    folder.mkdir(parents=True, exist_ok=True)
    QUAR_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(folder.glob(TXT_GLOB))
    if not candidates:
        print(f"• {folder}: no candidates")
        return 0, 0

    # map (group_key, hash) -> list[Path]
    buckets: Dict[Tuple[str, str], List[Path]] = {}
    for p in candidates:
        try:
            key = content_key(p, loose=loose)
        except Exception as e:
            print(f"  ⚠️  Skip {p.name}: {e}")
            continue
        buckets.setdefault(key, []).append(p)

    kept = 0
    moved = 0

    for (_group, _h), files in buckets.items():
        if len(files) == 1:
            kept += 1
            continue
        keep = pick_canonical(files)
        kept += 1
        dupes = [f for f in files if f != keep]
        if not dupes:
            continue
        if apply:
            for d in dupes:
                target = QUAR_DIR / folder.name / d.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(target))
            print(f"  🗂  {folder.name}: kept {keep.name}, quarantined {[d.name for d in dupes]}")
        else:
            print(f"  (dry-run) {folder.name}: would keep {keep.name}, quarantine {[d.name for d in dupes]}")
        moved += len(dupes)

    return kept, moved

def cross_prefer(kb_first: bool, a: Path, b: Path) -> Path:
    """Choose preferred path across folders (prefer KB if kb_first=True)."""
    if kb_first:
        return a if a.parent == KB_DIR else b
    return a if a.parent == INJECTED_DIR else b

def dedupe_cross_folder(*, loose: bool, apply: bool, prefer_kb: bool) -> Tuple[int, int]:
    """
    De-dupe ACROSS injected_outputs and scenario_runs:
    if two files have same canonical content, keep KB version (by default) and
    quarantine the duplicate from the other folder.
    """
    QUAR_DIR.mkdir(parents=True, exist_ok=True)
    all_files = list(INJECTED_DIR.glob(TXT_GLOB)) + list(KB_DIR.glob(TXT_GLOB))

    # map (group_key, hash) -> list[Path] across folders
    buckets: Dict[Tuple[str, str], List[Path]] = {}
    for p in all_files:
        try:
            key = content_key(p, loose=loose)
        except Exception as e:
            print(f"  ⚠️  Skip {p.name}: {e}")
            continue
        buckets.setdefault(key, []).append(p)

    kept = 0
    moved = 0

    for key, files in buckets.items():
        if len(files) <= 1:
            kept += len(files)
            continue

        # choose preferred folder copy first, then within that use pick_canonical
        kb_files = [f for f in files if f.parent == KB_DIR]
        inj_files = [f for f in files if f.parent == INJECTED_DIR]

        if prefer_kb and kb_files:
            keep = pick_canonical(kb_files)
        elif not prefer_kb and inj_files:
            keep = pick_canonical(inj_files)
        else:
            keep = pick_canonical(files)

        kept += 1
        dupes = [f for f in files if f != keep]
        if apply:
            for d in dupes:
                target = QUAR_DIR / d.parent.name / d.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(target))
            print(f"  🧹 cross: kept {keep.relative_to(ROOT)}, quarantined {[d.relative_to(ROOT) for d in dupes]}")
        else:
            print(f"  (dry-run) cross: would keep {keep.relative_to(ROOT)}, quarantine {[d.relative_to(ROOT) for d in dupes]}")
        moved += len(dupes)

    return kept, moved

def main():
    ap = argparse.ArgumentParser(description="De-duplicate scenario output files.")
    ap.add_argument("--apply", action="store_true", help="Perform moves (default is dry-run)")
    ap.add_argument("--loose", action="store_true", help="Ignore volatile lines (timestamps) and whitespace")
    ap.add_argument("--strict", action="store_true", help="Exact byte match only (default)")
    ap.add_argument("--cross", action="store_true", help="Also dedupe across KB and injected folders")
    ap.add_argument("--prefer", choices=["kb", "injected"], default="kb", help="When cross-deduping, which folder to keep")
    args = ap.parse_args()

    loose = args.loose and not args.strict

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode   : {mode}")
    print(f"Loose  : {loose}")
    print(f"Targets: {INJECTED_DIR} and {KB_DIR}")

    kept_i, moved_i = dedupe_folder(INJECTED_DIR, loose=loose, apply=args.apply)
    kept_k, moved_k = dedupe_folder(KB_DIR, loose=loose, apply=args.apply)

    if args.cross:
        kept_c, moved_c = dedupe_cross_folder(loose=loose, apply=args.apply, prefer_kb=(args.prefer == "kb"))
    else:
        kept_c = moved_c = 0

    print("\nSummary")
    print(f"  Kept        : {kept_i + kept_k + kept_c}")
    print(f"  Quarantined : {moved_i + moved_k + moved_c}")
    if not args.apply:
        print("\nℹ️  DRY-RUN. Re-run with --apply to make changes.")

if __name__ == "__main__":
    main()