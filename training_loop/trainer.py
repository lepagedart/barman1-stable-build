# training_loop/trainer.py
# ------------------------------------------------------------
# Lloyd Training Loop v4
# - Run scenarios -> Lloyd output (.txt)
# - Evaluate -> fill WLSHD + write metrics JSON + rolling CSV
# - Write a MANIFEST of only the files from this evaluation run
# - Gate -> inject/divert ONLY files from manifest (no globbing the whole folder)
# - No duplicate writes: skip if content hash is identical; only create _noXX for changed content
# - Diff reports only when a _noXX variant is created
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import csv
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import itertools
import hashlib

import requests
from dotenv import load_dotenv

load_dotenv()  # .env at project root

# ---------- Paths ----------
ROOT = Path(".").resolve()
TL = ROOT / "training_loop"

INPUT_DIR = TL / "scenario_inputs"
PROCESSED_INPUTS_DIR = TL / "processed_inputs"
STAGING_DIR = TL / "scenario_outputs"          # raw run outputs from Lloyd
EVAL_DIR = TL / "evaluator_outputs"            # evaluated, WLSHD-filled
EVAL_METRICS_DIR = EVAL_DIR / "metrics"        # metrics JSON lives here (master copy)
INJECTED_DIR = TL / "injected_outputs"         # evaluated files that were injected to KB
CRITIQUE_DIR = TL / "critique_only"            # evaluated files below gate
RUN_ARCHIVE_DIR = TL / "run_archive"           # optional archival of staging files
DIFF_DIR = TL / "diff_reports"                 # textual diffs across re-runs
MANIFEST_LATEST = EVAL_DIR / "last_manifest.txt"

KB_DIR = ROOT / "knowledge_base" / "training_modules" / "scenario_runs"
SCRIPTS_DIR = ROOT / "scripts"
TAG_SCRIPT_PATH = SCRIPTS_DIR / "update_tag_vocab.py"

# Rolling CSV (lives under evaluator_outputs/metrics by default)
SUMMARY_CSV = EVAL_METRICS_DIR / "summary_metrics.csv"

# ---------- Lloyd endpoint ----------
LLOYD_API_URL = "http://localhost:5000/"

# ---------- Evaluator prompts ----------
EVALUATOR_SYSTEM_PROMPT_WLSHD = (
    "You are the senior trainer for an AI beverage director named Lloyd.\n"
    "Write a direct, actionable critique titled EXACTLY: 'What Lloyd Should Have Done:' that:\n"
    "- Calls out missing pieces or overreach\n"
    "- Adds system/process solutions\n"
    "- Adds tone/structure guidance\n"
    "- Is concise (bullets ok), zero fluff, consultant-grade\n"
    "Return ONLY the body to replace [PLACEHOLDER], starting with that exact heading."
)

EVALUATOR_SYSTEM_PROMPT_METRICS = (
    "You are a strict evaluator. Read the scenario fields and Lloyd's response (and WLSHD if present), "
    "then score on a 1–5 scale where 5 = excellent client-ready, 1 = unacceptable. "
    "Return ONLY valid JSON with keys:\n"
    "{\n"
    '  "score_overall": number,\n'
    '  "clarity": number,\n'
    '  "feasibility": number,\n'
    '  "structure": number,\n'
    '  "actionability": number,\n'
    '  "alignment": number,\n'
    '  "notes": "short free text"\n'
    "}\n"
    "No markdown, no backticks—just JSON."
)

# ---------- Safe write / hashing helpers ----------

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _with_no_suffix(path: Path, n: int) -> Path:
    """
    Insert _noXX before the final suffix(s).
    e.g. scenario_041_output.txt -> scenario_041_output_no01.txt
         scenario_041_output.metrics.json -> scenario_041_output_no01.metrics.json
    """
    suffixes = "".join(path.suffixes)  # ".txt" or ".metrics.json"
    base = path.name[: -len(suffixes)] if suffixes else path.name
    return path.with_name(f"{base}_no{n:02d}{suffixes}")

def unique_path(path: Path) -> Path:
    """Return a non-colliding path by adding _no01, _no02, … if needed."""
    if not path.exists():
        return path
    for i in itertools.count(1):
        cand = _with_no_suffix(path, i)
        if not cand.exists():
            return cand
    return path  # fallback

def write_if_changed(path: Path, text: str) -> Path:
    """
    - If `path` exists with identical content, return `path` (no write, no _noXX).
    - If `path` exists but content differs, write to a unique _noXX path.
    - If `path` doesn't exist, write to `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_hash = _sha256_text(text)

    if path.exists():
        old_hash = _sha256_file(path)
        if old_hash == new_hash:
            return path  # identical; do nothing
        target = unique_path(path)  # content differs -> create _noXX
        target.write_text(text, encoding="utf-8")
        return target

    # fresh write
    path.write_text(text, encoding="utf-8")
    return path

# ---------- Utilities ----------

def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    CRITIQUE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _slug(n: int) -> str:
    return f"{n:03d}"

def _scenario_out_name(n: int) -> str:
    return f"scenario_{_slug(n)}_output.txt"

def _scenario_metrics_name(n: int) -> str:
    return f"scenario_{_slug(n)}_metrics.json"

def _sibling_metrics_name(txt_name: str) -> str:
    # scenario_XXX_output.txt -> scenario_XXX_output.metrics.json
    return txt_name.replace(".txt", ".metrics.json")

def load_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for p in sorted(INPUT_DIR.glob("*.json")):
        try:
            scenarios.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping {p.name}: JSON error {e}")
    return scenarios

def move_input_to_processed(p: Path) -> None:
    # Moves consumed input JSON into processed_inputs with timestamp suffix
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = PROCESSED_INPUTS_DIR / f"{p.stem}__{stamp}{p.suffix}"
        p.replace(dest)
    except Exception as e:
        print(f"⚠️  Could not move input {p.name} → processed_inputs: {e}")

def check_lloyd_health(url: str = LLOYD_API_URL) -> bool:
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            print(f"✅ Lloyd is up at {url}")
            return True
        print(f"⚠️ Lloyd responded with status {r.status_code} at {url}")
        return False
    except Exception as e:
        print(f"❌ Lloyd not reachable at {url} — {e}")
        return False

def send_prompt_to_lloyd(venue: str, user_prompt: str) -> str:
    payload = {
        "venue_concept": venue,
        "user_prompt": user_prompt,
        "use_live_search": False,
    }
    try:
        r = requests.post(LLOYD_API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return (data.get("assistant_response") or r.text or "").strip()
    except Exception as e:
        return f"[ERROR] Failed to reach Lloyd: {e}"

def format_output(scenario: Dict[str, Any], lloyd_response: str) -> str:
    out: List[str] = []
    out.append(f"Title: {scenario['title']}")
    out.append(f"Tags: {', '.join(scenario['tags'])}")
    out.append("Scenario Type: Training")
    out.append(f"System Mod: {scenario['system_mod']}")
    out.append(f"Venue Context: {scenario['venue_context']}")
    out.append(f"Prompt: {scenario['prompt']}")
    out.append("Lloyd's Response:")
    out.append(lloyd_response)
    out.append("\nWhat Lloyd Should Have Done:\n[PLACEHOLDER]\n")
    out.append(f"(Generated on { _ts() })")
    return "\n\n".join(out)

def save_txt(directory: Path, scenario_number: int, text: str) -> Path:
    return write_if_changed(directory / _scenario_out_name(scenario_number), text)

def extract_fields_from_txt(content: str) -> Dict[str, str]:
    """
    Parses our formatted .txt into a dict of blocks:
    Title, Tags, Scenario Type, System Mod, Venue Context, Prompt, Lloyd's Response, What Lloyd Should Have Done
    """
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    header_re = re.compile(r"^[A-Za-z][A-Za-z ']+:\s*$")
    for line in content.splitlines():
        stripped = line.strip()
        if header_re.match(stripped):
            current = stripped[:-1]
            fields[current] = ""
        elif current:
            fields[current] += line + "\n"
    for k in list(fields.keys()):
        fields[k] = fields[k].rstrip()
    return fields

# ---------- OpenAI Helpers ----------

def _openai_available() -> Tuple[bool, Optional[str]]:
    try:
        import openai  # noqa: F401
    except Exception as e:
        return False, f"OpenAI import failed: {e}"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set in environment"
    return True, None

def evaluate_wlshd(content: str, model: str) -> str:
    ok, err = _openai_available()
    if not ok:
        return f"[ERROR] {err}"
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

    fields = extract_fields_from_txt(content)
    payload = "\n".join(
        [
            f"Scenario Type: {fields.get('Scenario Type', '')}",
            f"Tags: {fields.get('Tags', '')}",
            f"System Mod: {fields.get('System Mod', '')}",
            f"Venue Context: {fields.get('Venue Context', '')}",
            "",
            f"Prompt: {fields.get('Prompt', '')}",
            "",
            "Lloyd's Response:",
            fields.get("Lloyd's Response", ""),
        ]
    )
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT_WLSHD},
                {"role": "user", "content": payload},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else "[ERROR] No response from OpenAI"
    except Exception as e:
        return f"[ERROR] OpenAI call failed: {e}"

def evaluate_metrics(content: str, model: str) -> Dict[str, Any]:
    """
    Returns a metrics dict and guarantees JSON is created even on error.
    Keys (on success): score_overall, clarity, feasibility, structure, actionability, alignment, notes
    On error: score_overall = 0.0 and 'error' message.
    """
    base: Dict[str, Any] = {
        "score_overall": 0.0,
        "clarity": 0.0,
        "feasibility": 0.0,
        "structure": 0.0,
        "actionability": 0.0,
        "alignment": 0.0,
        "notes": "",
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    ok, err = _openai_available()
    if not ok:
        base["error"] = err
        return base

    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

    fields = extract_fields_from_txt(content)
    payload = "\n".join(
        [
            f"Title: {fields.get('Title', '')}",
            f"Tags: {fields.get('Tags', '')}",
            f"System Mod: {fields.get('System Mod', '')}",
            f"Venue Context: {fields.get('Venue Context', '')}",
            "",
            "Prompt:",
            fields.get("Prompt", ""),
            "",
            "Lloyd's Response:",
            fields.get("Lloyd's Response", ""),
            "",
            "What Lloyd Should Have Done:",
            fields.get("What Lloyd Should Have Done", ""),
        ]
    )

    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT_METRICS},
                {"role": "user", "content": payload},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        for k in ["score_overall", "clarity", "feasibility", "structure", "actionability", "alignment", "notes"]:
            if k not in data:
                data[k] = base[k]
        data["scored_at"] = base["scored_at"]
        return data
    except Exception as e:
        base["error"] = f"OpenAI metrics failed: {e}"
        return base

# ---------- Diff helpers ----------

def write_diff_report(old_file: Path, new_file: Path) -> None:
    """
    Compare old_file vs. new_file and save a diff into diff_reports/.
    """
    import difflib
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    try:
        old_text = old_file.read_text(encoding="utf-8").splitlines(keepends=True)
        new_text = new_file.read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception as e:
        print(f"⚠️ Could not read for diff: {old_file}, {new_file} ({e})")
        return

    diff = difflib.unified_diff(
        old_text, new_text,
        fromfile=str(old_file.name),
        tofile=str(new_file.name),
        lineterm=""
    )
    report_name = f"{old_file.stem}__vs__{new_file.stem}.diff"
    report_path = DIFF_DIR / report_name
    report_path.write_text("\n".join(diff), encoding="utf-8")
    print(f"📝 Diff report written → {report_path}")

# ---------- Evaluation runners ----------

def run_evaluator(src_dir: Path, dest_dir: Path, model_wlshd: str, model_metrics: str, inplace: bool = False) -> Path:
    """
    Evaluate all .txt in src_dir and write to dest_dir (or copy-in-place _noXX).
    Returns the path to a manifest (list of evaluated basenames) for this run.
    """
    files = sorted(src_dir.glob("scenario_*_output.txt"))
    print(f"🔎 Evaluator scanning: {src_dir}  |  {len(files)} file(s)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not files:
        print("⚠️  No .txt files found to evaluate (run --run first).")
        # still write an empty manifest
        manifest = dest_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        manifest.write_text("", encoding="utf-8")
        MANIFEST_LATEST.write_text("", encoding="utf-8")
        return manifest

    processed_names: List[str] = []
    for p in files:
        content = p.read_text(encoding="utf-8")

        # 1) Fill WLSHD if needed
        if "[PLACEHOLDER]" in content:
            critique = evaluate_wlshd(content, model=model_wlshd)
            content = content.replace("[PLACEHOLDER]", critique)
            content += f"\n\n(Evaluated on { _ts() })"
        else:
            if "(Evaluated on " not in content:
                content += f"\n\n(Evaluated on { _ts() })"

        # 2) Write evaluated file (no duplicate writes)
        if inplace:
            # copy-in-place semantics with change detection
            out_path = write_if_changed(p, content)
            if out_path.name != p.name:
                # we created a _noXX – compare against base and write diff
                write_diff_report(p, out_path)
            print(f"💬 Evaluated (in-place -> {out_path.name})")
        else:
            out_path = write_if_changed(dest_dir / p.name, content)
            if out_path.name != (dest_dir / p.name).name:
                # base existed with different content -> compare to base in dest_dir if present
                base_file = dest_dir / p.name
                if base_file.exists():
                    write_diff_report(base_file, out_path)
            print(f"💬 Evaluated → {out_path}")

        processed_names.append(out_path.name)

        # 3) Always create metrics JSON (even if later gated)
        metrics = evaluate_metrics(content, model=model_metrics)

        # Master metrics copy under evaluator_outputs/metrics
        scenario_num = _extract_scenario_number_from_name(p.name)
        m_master_base = (
            EVAL_METRICS_DIR / _scenario_metrics_name(scenario_num)
            if scenario_num is not None
            else EVAL_METRICS_DIR / (p.stem + ".metrics.json")
        )
        m_master_base.parent.mkdir(parents=True, exist_ok=True)
        # Write metrics; if identical, skip dup; if changed, _noXX
        m_master = write_if_changed(m_master_base, json.dumps(metrics, indent=2))

        # Sibling copy adjacent to the evaluated .txt
        _ = write_if_changed(Path(out_path.with_suffix(".metrics.json")), json.dumps(metrics, indent=2))

        # 4) Append to rolling CSV
        _append_metrics_csv(m_master, metrics, source_txt=out_path)

    # 5) Write MANIFEST for this evaluation run
    manifest = dest_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    manifest.write_text("\n".join(processed_names) + ("\n" if processed_names else ""), encoding="utf-8")
    MANIFEST_LATEST.write_text(manifest.name, encoding="utf-8")
    print(f"🗂️  Wrote manifest → {manifest}  (and updated {MANIFEST_LATEST})")
    return manifest

def _extract_scenario_number_from_name(name: str) -> Optional[int]:
    m = re.match(r"scenario_(\d{3})_output\.txt", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _append_metrics_csv(metrics_path: Path, metrics: Dict[str, Any], source_txt: Path) -> None:
    """
    Append a row to SUMMARY_CSV with columns:
    ts, scenario_file, score_overall, clarity, feasibility, structure, actionability, alignment, notes
    """
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario_file": source_txt.name,
        "metrics_file": str(metrics_path.name),
        "score_overall": metrics.get("score_overall", 0.0),
        "clarity": metrics.get("clarity", 0.0),
        "feasibility": metrics.get("feasibility", 0.0),
        "structure": metrics.get("structure", 0.0),
        "actionability": metrics.get("actionability", 0.0),
        "alignment": metrics.get("alignment", 0.0),
        "notes": (metrics.get("notes") or "")[:500],
        "error": metrics.get("error", ""),
    }

    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ts", "scenario_file", "metrics_file",
                "score_overall", "clarity", "feasibility", "structure", "actionability", "alignment",
                "notes", "error"
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(row)

# ---------- KB + gating helpers ----------

def _read_score_for(p: Path) -> Optional[float]:
    """Find score_overall for evaluated text `p` via sibling or metrics dir."""
    # 1) sibling metrics
    sib = p.with_suffix(".metrics.json")
    if sib.exists():
        try:
            data = json.loads(sib.read_text(encoding="utf-8"))
            return float(data.get("score_overall"))
        except Exception:
            pass

    # 2) metrics dir by scenario number
    num = _extract_scenario_number_from_name(p.name)
    if num is not None:
        cand = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                return float(data.get("score_overall"))
            except Exception:
                pass

    # 3) metrics dir sibling-named
    cand2 = EVAL_METRICS_DIR / _sibling_metrics_name(p.name)
    if cand2.exists():
        try:
            data = json.loads(cand2.read_text(encoding="utf-8"))
            return float(data.get("score_overall"))
        except Exception:
            pass

    return None

def _copy_metrics_for(p: Path, dest_dir: Path) -> None:
    """Copy whichever metrics file exists for text p into dest_dir (no duplicate writes)."""
    # sibling
    sib = p.with_suffix(".metrics.json")
    if sib.exists():
        write_if_changed(dest_dir / sib.name, sib.read_text(encoding="utf-8"))
        return

    # metrics dir by number
    num = _extract_scenario_number_from_name(p.name)
    if num is not None:
        cand = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if cand.exists():
            write_if_changed(dest_dir / cand.name, cand.read_text(encoding="utf-8"))
            return

    # metrics dir sibling name
    cand2 = EVAL_METRICS_DIR / _sibling_metrics_name(p.name)
    if cand2.exists():
        write_if_changed(dest_dir / cand2.name, cand2.read_text(encoding="utf-8"))
        return

def _load_manifest_list(path: Optional[Path]) -> Optional[List[str]]:
    """
    Accept either:
      - a manifest file that lists evaluated basenames (one per line), OR
      - a file containing just the manifest filename (MANIFEST_LATEST).
    Returns list of basenames or None if not available.
    """
    if not path:
        return None
    if not path.exists():
        return None

    # If pointing at last_manifest.txt (which contains a single manifest filename)
    try:
        if path.name == "last_manifest.txt":
            manifest_name = path.read_text(encoding="utf-8").strip()
            if not manifest_name:
                return None
            real_manifest = path.parent / manifest_name
            if real_manifest.exists():
                return [ln.strip() for ln in real_manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return None
    except Exception:
        return None

    # Else: treat as a normal manifest with names
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return None

# ---------- Metrics selection helpers (keep only the latest next to injected .txt) ----------

from datetime import datetime
from typing import Tuple

def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        # tolerate "Z" suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _read_metrics_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _metrics_candidates_for_txt(txt_path: Path) -> List[Tuple[Path, Dict[str, Any], datetime]]:
    """
    Return all available metrics variants for a given evaluated txt, each as (path, data, score_dt).
    score_dt is from JSON['scored_at'] when present, else file mtime.
    """
    cands: List[Tuple[Path, Dict[str, Any], datetime]] = []

    # 1) sibling next to the evaluated file (common case)
    sib = txt_path.with_suffix(".metrics.json")
    if sib.exists():
        data = _read_metrics_json(sib)
        if data is not None:
            dt = _parse_iso(str(data.get("scored_at") or "")) or datetime.fromtimestamp(sib.stat().st_mtime)
            cands.append((sib, data, dt))

    # 2) metrics dir by scenario number
    num = _extract_scenario_number_from_name(txt_path.name)
    if num is not None:
        pnum = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if pnum.exists():
            data = _read_metrics_json(pnum)
            if data is not None:
                dt = _parse_iso(str(data.get("scored_at") or "")) or datetime.fromtimestamp(pnum.stat().st_mtime)
                cands.append((pnum, data, dt))

    # 3) metrics dir sibling-named
    pstem = EVAL_METRICS_DIR / _sibling_metrics_name(txt_path.name)
    if pstem.exists():
        data = _read_metrics_json(pstem)
        if data is not None:
            dt = _parse_iso(str(data.get("scored_at") or "")) or datetime.fromtimestamp(pstem.stat().st_mtime)
            cands.append((pstem, data, dt))

    # dedupe identical paths
    seen = set()
    uniq: List[Tuple[Path, Dict[str, Any], datetime]] = []
    for t in cands:
        if t[0] not in seen:
            uniq.append(t)
            seen.add(t[0])
    return uniq

def _pick_latest_metrics(txt_path: Path) -> Optional[Tuple[Dict[str, Any], Path]]:
    """
    Choose the newest metrics for txt_path. Returns (data, source_path) or None.
    """
    cands = _metrics_candidates_for_txt(txt_path)
    if not cands:
        return None
    cands.sort(key=lambda x: x[2])  # by datetime
    best_path, best_data, _ = cands[-1][0], cands[-1][1], cands[-1][2]
    return best_data, best_path

def _write_json_atomic(path: Path, data: Dict[str, Any]) -> Path:
    """
    Overwrite (atomically) JSON at `path`. Unlike safe_write, this keeps exactly one file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX
    return path



def inject_to_kb(from_dir: Path, gate: float, only_names: Optional[List[str]] = None) -> None:
    """
    Inject only txts in `from_dir`. For each injected txt, keep exactly ONE metrics json alongside it in the KB:
    the newest one (by scored_at or mtime). This file is overwritten atomically each run.
    """
    files = sorted(from_dir.glob("scenario_*_output.txt"))
    injected = 0
    diverted = 0

    for p in files:
        txt = p.read_text(encoding="utf-8")
        if "[PLACEHOLDER]" in txt:
            print(f"⏭️  Skipping (no WLSHD yet): {p.name}")
            continue

        score = _read_score_for(p)
        if score is None:
            print(f"🧯 Diverted (no/invalid metrics): {p.name}")
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            # keep any metrics copy in critique_only for traceability (non-overwriting)
            c_pick = _pick_latest_metrics(p)
            if c_pick:
                write_if_changed((CRITIQUE_DIR / dest.name).with_suffix(".metrics.json"),
                                 json.dumps(c_pick[0], indent=2))
            diverted += 1
            continue

        if score >= gate:
            # 1) Inject text (preserve history of content)
            kb_txt_path = write_if_changed(KB_DIR / p.name, txt)

            # 2) Select newest metrics and keep ONE canonical copy next to the injected txt
            pick = _pick_latest_metrics(p)
            if pick:
                kb_metrics_path = kb_txt_path.with_suffix(".metrics.json")
                _write_json_atomic(kb_metrics_path, pick[0])  # overwrite to keep only latest

            # 3) Optional: audit copy in injected_outputs (non-overwriting history)
            inj_txt = write_if_changed(INJECTED_DIR / p.name, txt)
            if pick:
                write_if_changed(inj_txt.with_suffix(".metrics.json"), json.dumps(pick[0], indent=2))

            print(f"📚 Injected (score={score:.2f}) → {kb_txt_path.name}  (+ latest metrics)")
            injected += 1
        else:
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            # stash metrics alongside diverted file for review
            pick = _pick_latest_metrics(p)
            if pick:
                write_if_changed(dest.with_suffix(".metrics.json"), json.dumps(pick[0], indent=2))
            print(f"🧯 Diverted (score={score:.2f} < gate {gate:.2f}) → {dest.name}")
            diverted += 1

    print(f"📦 Results: Injected={injected}  Diverted={diverted}")
    files = sorted(from_dir.glob("scenario_*_output.txt"))
    injected = 0
    diverted = 0

    for p in files:
        txt = p.read_text(encoding="utf-8")
        if "[PLACEHOLDER]" in txt:
            print(f"⏭️  Skipping (no WLSHD yet): {p.name}")
            continue

        score = _read_score_for(p)
        if score is None:
            print(f"🧯 Diverted (no/invalid metrics): {p.name}")
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            _copy_metrics_for(p, CRITIQUE_DIR)  # keep metrics with critique copy
            diverted += 1
            continue

        if score >= gate:
            # 1) Inject text to KB (🛑 do NOT copy metrics into KB anymore)
            kb_target = write_if_changed(KB_DIR / p.name, txt)

            # 2) Audit copy (text + metrics) into injected_outputs
            inj_target = write_if_changed(INJECTED_DIR / p.name, txt)
            _copy_metrics_for(p, INJECTED_DIR)

            print(f"📚 Injected (score={score:.2f}) → KB & injected_outputs: {kb_target.name}")
            injected += 1
        else:
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            _copy_metrics_for(p, CRITIQUE_DIR)
            print(f"🧯 Diverted (score={score:.2f} < gate {gate:.2f}) → critique_only: {dest.name}")
            diverted += 1

    print(f"📦 Results: Injected={injected}  Diverted={diverted}")
    # Inject/divert ONLY files whose names are listed in only_names (manifest).
    # If only_names is None, it will fallback to scanning all files in from_dir.
    candidates = sorted(from_dir.glob("scenario_*_output.txt"))
    files = [p for p in candidates if (not only_names or p.name in only_names)]

    injected = 0
    diverted = 0

    if only_names is not None:
        print(f"🧭 Inject scope: {len(files)}/{len(candidates)} files (manifest-limited)")
    else:
        print(f"🧭 Inject scope: {len(files)} files (full folder scan)")

    for p in files:
        txt = p.read_text(encoding="utf-8")
        if "[PLACEHOLDER]" in txt:
            print(f"⏭️  Skipping (no WLSHD yet): {p.name}")
            continue

        score = _read_score_for(p)
        if score is None:
            print(f"🧯 Diverted (no/invalid metrics): {p.name}")
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            _copy_metrics_for(p, CRITIQUE_DIR)
            diverted += 1
            continue

        if score >= gate:
            # 1) Inject to KB (skip if identical content)
            kb_target = write_if_changed(KB_DIR / p.name, txt)
            _copy_metrics_for(p, KB_DIR)
            # 2) Audit copy
            inj_target = write_if_changed(INJECTED_DIR / p.name, txt)
            _copy_metrics_for(p, INJECTED_DIR)
            print(f"📚 Injected (score={score:.2f}) → KB & injected_outputs: {kb_target.name}")
            injected += 1
        else:
            dest = write_if_changed(CRITIQUE_DIR / p.name, txt)
            _copy_metrics_for(p, CRITIQUE_DIR)
            print(f"🧯 Diverted (score={score:.2f} < gate {gate:.2f}) → critique_only: {dest.name}")
            diverted += 1

    print(f"📦 Results: Injected={injected}  Diverted={diverted}")

def rebuild_vectorstore() -> None:
    print("🔁 Rebuilding vectorstore...")
    subprocess.run(["python", "kb_loader.py", "--rebuild"], check=True)
    if TAG_SCRIPT_PATH.exists():
        print("🏷️ Updating tag vocab…")
        subprocess.run(["python", str(TAG_SCRIPT_PATH)], check=True)
    else:
        print("ℹ️  Tag vocab script not found; skipping.")

# ---------- Runner: Run -> Evaluate -> Inject -> Rebuild ----------

def _run_scenarios() -> None:
    scenarios = load_scenarios()
    if not scenarios:
        print("⚠️  No scenarios found in scenario_inputs")
        return

    for s in scenarios:
        try:
            n = int(s["scenario_number"])
        except Exception:
            try:
                n = int(str(s["scenario_number"]).lstrip("0") or "0")
            except Exception:
                print(f"⚠️  Invalid scenario_number in one file; skipping")
                continue

        print(f"🚀 Scenario {n:03d} — {s['title']}")
        resp = send_prompt_to_lloyd(s["venue_context"], s["prompt"])
        txt = format_output(s, resp)
        out_path = save_txt(STAGING_DIR, n, txt)
        print(f"✅ Saved → {out_path}")

        # move consumed input JSON → processed_inputs
        for cand in INPUT_DIR.glob(f"scenario_{n:03d}.json"):
            move_input_to_processed(cand)

def _evaluate(args) -> Path:
    manifest = run_evaluator(
        src_dir=Path(args.eval_src),
        dest_dir=Path(args.eval_dest),
        model_wlshd=args.eval_model,
        model_metrics=args.metrics_model or args.eval_model,
        inplace=args.eval_inplace,
    )
    return manifest

def _inject(args, manifest_path: Optional[Path]) -> None:
    only_names = _load_manifest_list(manifest_path) if manifest_path else None
    inject_to_kb(Path(args.inject_src), gate=float(args.gate), only_names=only_names)

def _rebuild(_args) -> None:
    rebuild_vectorstore()

# ---------- CLI ----------

def main() -> None:
    ensure_dirs()

    ap = argparse.ArgumentParser(
        description="Lloyd trainer + evaluator + metrics + gated KB injection (manifest-scoped)"
    )

    # Main stages
    ap.add_argument("--run", action="store_true", help="Run scenarios from scenario_inputs → scenario_outputs")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate .txt (fill WLSHD + write metrics JSON/CSV + manifest)")
    ap.add_argument("--inject", action="store_true", help="Inject/divert ONLY files from the current manifest (or --manifest)")
    ap.add_argument("--rebuild", action="store_true", help="After inject, rebuild vectorstore and tag vocab")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run → Evaluate → Inject → Rebuild in one shot (uses defaults for paths/models)",
    )

    # Paths / options
    ap.add_argument("--eval-src", default=str(STAGING_DIR), help="Where to read .txt for evaluation (default: scenario_outputs)")
    ap.add_argument("--eval-dest", default=str(EVAL_DIR), help="Where to write evaluated .txt (default: evaluator_outputs)")
    ap.add_argument("--eval-inplace", action="store_true", help="Write WLSHD back into source files instead of dest folder")
    ap.add_argument("--inject-src", default=str(EVAL_DIR), help="Folder to read evaluated files from for KB injection")
    ap.add_argument("--manifest", default=str(MANIFEST_LATEST),
                    help="Path to manifest or last_manifest.txt (default: evaluator_outputs/last_manifest.txt)")

    # Models
    ap.add_argument("--eval-model", default="gpt-4o", help="Model for WLSHD critique (default: gpt-4o)")
    ap.add_argument("--metrics-model", default=None, help="Model for metrics JSON (default: uses --eval-model if not set)")

    # Gate
    ap.add_argument("--gate", type=float, default=4.0, help="Minimum score_overall to inject into KB (default: 4.0)")

    args = ap.parse_args()

    # Guard: if running scenarios, ensure Lloyd is reachable
    if args.run:
        if not check_lloyd_health(LLOYD_API_URL):
            print("🚫 Aborting: Lloyd must be running (python app.py) before using --run")
            return

    # One-shot pipeline
    manifest_path: Optional[Path] = None
    if args.all:
        args.run = True
        args.evaluate = True
        args.inject = True
        args.rebuild = True
        args.eval_src = str(STAGING_DIR)
        args.eval_dest = str(EVAL_DIR)
        args.inject_src = str(EVAL_DIR)
        args.manifest = str(MANIFEST_LATEST)

    if args.run:
        _run_scenarios()

    if args.evaluate:
        manifest_path = _evaluate(args)

    if args.inject:
        # If user didn't just evaluate, fall back to --manifest or last_manifest.txt
        if manifest_path is None and args.manifest:
            manifest_path = Path(args.manifest)
        _inject(args, manifest_path)

    if args.rebuild:
        _rebuild(args)

if __name__ == "__main__":
    main()