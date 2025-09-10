# training_loop/trainer.py
# ------------------------------------------------------------
# Lloyd Training Loop v3.2 (scoped eval/inject + manifests)
# - Run scenarios -> Lloyd output (.txt) [scoped list saved]
# - Evaluate -> fill WLSHD + write metrics JSON + CSV (scoped)
# - Gate -> inject to KB if score_overall >= gate (scoped)
# - Diff reports only for the scoped files
# - Safe write everywhere (_noXX) to avoid clobber
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import csv
import json
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()  # .env at project root

# ---------- safe write helpers ----------

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

def safe_write(path: Path, text: str) -> Path:
    """Write to `path` or a unique variant if it exists; return the actual path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = unique_path(path)
    target.write_text(text, encoding="utf-8")
    return target

# ---------- Paths ----------
ROOT = Path(".").resolve()
TL = ROOT / "training_loop"

INPUT_DIR = TL / "scenario_inputs"
PROCESSED_INPUTS_DIR = TL / "processed_inputs"
STAGING_DIR = TL / "scenario_outputs"         # raw run outputs from Lloyd
EVAL_DIR = TL / "evaluator_outputs"           # evaluated, WLSHD-filled
EVAL_METRICS_DIR = EVAL_DIR / "metrics"       # metrics JSON lives here (master copy)
INJECTED_DIR = TL / "injected_outputs"        # evaluated files that were injected to KB
CRITIQUE_DIR = TL / "critique_only"           # evaluated files below gate
RUN_ARCHIVE_DIR = TL / "run_archive"          # optional archival of staging files
DIFF_DIR = TL / "diff_reports"

# KB
KB_DIR = ROOT / "knowledge_base" / "training_modules" / "scenario_runs"
SCRIPTS_DIR = ROOT / "scripts"
TAG_SCRIPT_PATH = SCRIPTS_DIR / "update_tag_vocab.py"

# Manifests (so later stages know the exact files from the last stage)
LAST_RUN_MANIFEST  = TL / "last_run_outputs.json"     # names of scenario_*_output.txt created in --run
LAST_EVAL_MANIFEST = TL / "last_eval_outputs.json"    # names of evaluated files written in --evaluate

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

# ---------- Utilities ----------

def ensure_dirs() -> None:
    for d in [
        INPUT_DIR, PROCESSED_INPUTS_DIR, STAGING_DIR, EVAL_DIR, EVAL_METRICS_DIR,
        INJECTED_DIR, CRITIQUE_DIR, RUN_ARCHIVE_DIR, KB_DIR, DIFF_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)

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

def _save_manifest(path: Path, names: List[str]) -> None:
    path.write_text(
        json.dumps({"files": names, "saved_at": datetime.utcnow().isoformat() + "Z"}, indent=2),
        encoding="utf-8",
    )

def _load_manifest(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("files") or [])
    except Exception:
        return []

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
    return safe_write(directory / _scenario_out_name(scenario_number), text)

def extract_fields_from_txt(content: str) -> Dict[str, str]:
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
        return text if text else "[ERROR] No response from OpenAI]"
    except Exception as e:
        return f"[ERROR] OpenAI call failed: {e}"

def evaluate_metrics(content: str, model: str) -> Dict[str, Any]:
    """
    Returns a metrics dict and guarantees JSON is created even on error.
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
    stamp = datetime.utcnow().isoformat(timespec="seconds")
    report_name = f"{old_file.stem}_vs_{new_file.stem}.diff"
    report_path = DIFF_DIR / report_name
    report_path.write_text(f"# (Evaluated on {stamp})\n" + "\n".join(diff), encoding="utf-8")
    print(f"📝 Diff report written → {report_path}")

# ---------- Evaluation runners ----------

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

def run_evaluator(
    src_dir: Path,
    dest_dir: Path,
    model_wlshd: str,
    model_metrics: str,
    inplace: bool = False,
    only_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Evaluate scenario_*_output.txt files.

    Behavior:
      - If only_names is provided, evaluate ONLY those filenames.
      - Else, if last_run_outputs.json exists, evaluate ONLY that last batch.
      - Else, evaluate everything in src_dir.
      - Fill WLSHD if [PLACEHOLDER] exists.
      - Always write metrics (master copy + sibling).
      - Never clobber: uses safe_write().
      - Returns list of evaluated basenames.
    """
    # Default scope from last run manifest if caller didn’t pass --only-names
    if only_names is None:
        last_run = _load_manifest(LAST_RUN_MANIFEST)  # returns List[str] or []
        only_names = last_run if last_run else None

    # Discover candidate files and apply filter
    candidates = sorted(src_dir.glob("scenario_*_output.txt"))
    name_set = {n.strip() for n in only_names} if only_names else None
    files = [p for p in candidates] if name_set is None else [p for p in candidates if p.name.strip() in name_set]

    print(f"🔎 Evaluator scanning: {src_dir}  |  matched {len(files)} file(s)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not files:
        print("⚠️  No matching .txt files found to evaluate.")
        return []

    produced: List[str] = []

    for p in files:
        content = p.read_text(encoding="utf-8")

        # Fill WLSHD if needed
        if "[PLACEHOLDER]" in content:
            critique = evaluate_wlshd(content, model=model_wlshd)
            content = content.replace("[PLACEHOLDER]", critique)
            content += f"\n\n(Evaluated on { _ts() })"
        elif "(Evaluated on " not in content:
            content += f"\n\n(Evaluated on { _ts() })"

        # Write evaluated file (safe, non-clobber)
        if inplace:
            out_path = safe_write(p, content)
            print(f"💬 Evaluated (in-place copy): {out_path.name}")
        else:
            out_path = safe_write(dest_dir / p.name, content)
            print(f"💬 Evaluated → {out_path.name}")

        produced.append(out_path.name)

        # Metrics (master + sibling)
        metrics = evaluate_metrics(content, model=model_metrics)
        scen_num = _extract_scenario_number_from_name(p.name)
        if scen_num is not None:
            m_master = safe_write(
                EVAL_METRICS_DIR / _scenario_metrics_name(scen_num),
                json.dumps(metrics, indent=2),
            )
        else:
            m_master = safe_write(
                EVAL_METRICS_DIR / (p.stem + ".metrics.json"),
                json.dumps(metrics, indent=2),
            )

        _ = safe_write(out_path.with_suffix(".metrics.json"), json.dumps(metrics, indent=2))
        _append_metrics_csv(m_master, metrics, source_txt=out_path)

        # Diff vs base (if this is a _noXX file and the base exists)
        base_name = out_path.name
        if "_no" in base_name:
            base_name = base_name.split("_no")[0] + ".txt"
        base_file = out_path.parent / base_name
        if base_file.exists() and base_file != out_path:
            write_diff_report(base_file, out_path)

    # Persist what we just evaluated (so next run can “only last batch” by default)
    _save_manifest(LAST_EVAL_MANIFEST, produced)

    return produced

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
    """Copy whichever metrics file exists for text p into dest_dir (no overwrite)."""
    # sibling
    sib = p.with_suffix(".metrics.json")
    if sib.exists():
        safe_write(dest_dir / sib.name, sib.read_text(encoding="utf-8"))
        return

    # metrics dir by number
    num = _extract_scenario_number_from_name(p.name)
    if num is not None:
        cand = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if cand.exists():
            safe_write(dest_dir / cand.name, cand.read_text(encoding="utf-8"))
            return

    # metrics dir sibling name
    cand2 = EVAL_METRICS_DIR / _sibling_metrics_name(p.name)
    if cand2.exists():
        safe_write(dest_dir / cand2.name, cand2.read_text(encoding="utf-8"))
        return

def inject_to_kb(from_dir: Path, gate: float, only_names: Optional[List[str]] = None) -> None:
    # Scope to only_names if provided
    candidates = sorted(from_dir.glob("scenario_*_output.txt"))
    files = [p for p in candidates if (not only_names or p.name in only_names)]

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
            safe_write(CRITIQUE_DIR / p.name, txt)
            _copy_metrics_for(p, CRITIQUE_DIR)
            diverted += 1
            continue

        if score >= gate:
            # Inject text only; DO NOT copy metrics to KB (keep KB lean)
            kb_target = safe_write(KB_DIR / p.name, txt)
            # Audit copy
            safe_write(INJECTED_DIR / p.name, txt)
            _copy_metrics_for(p, INJECTED_DIR)
            print(f"📚 Injected (score={score:.2f}) → KB & injected_outputs: {kb_target.name}")
            injected += 1
        else:
            dest = safe_write(CRITIQUE_DIR / p.name, txt)
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

# ---------- Question-only heuristics ----------

QUESTION_RE = re.compile(r"\?\s*(?:$|[\n•-])")

def looks_like_question_only(text: str) -> bool:
    # Heuristic: lots of question marks, no obvious plan sections
    q = len(QUESTION_RE.findall(text))
    has_plan = any(h in text.lower() for h in ["proposed plan", "action plan", "next steps", "implementation", "station map", "recipe", "checklist"])
    has_bullets = "•" in text or "-" in text[:500]
    return q >= 2 and not has_plan and not has_bullets

DEFAULT_ASSUMPTIONS = {
    "venue_type": "high-volume restaurant bar with 2 wells",
    "peak_volume": "120 covers/hour, ~180 drinks/hour",
    "staffing": "2 bartenders + 1 barback at peak",
    "menu": "10 cocktail core list, 4 beer drafts, 12 wines by the glass",
}

def build_assumption_reply(scenario: Dict[str, Any]) -> str:
    # Optional: allow scenarios to include 'auto_context' dict to override defaults
    ac = scenario.get("auto_context") or {}
    merged = {**DEFAULT_ASSUMPTIONS, **ac}
    lines = [
        "Thanks—proceed using temporary assumptions:",
        f"- Venue: {merged['venue_type']}",
        f"- Peak volume: {merged['peak_volume']}",
        f"- Staffing: {merged['staffing']}",
        f"- Menu: {merged['menu']}",
        "",
        "Please deliver the full recommendation based on these assumptions. "
        "Clearly label an **Assumptions** section up front."
    ]
    return "\n".join(lines)

# ---------- Runner: Run -> Evaluate -> Inject -> Rebuild ----------

def _run_scenarios() -> List[str]:
    scenarios = load_scenarios()
    if not scenarios:
        print("⚠️  No scenarios found in scenario_inputs")
        return []

    produced_names: List[str] = []

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

        # Auto-follow-up if Lloyd mainly asked questions
        if looks_like_question_only(resp):
            follow = build_assumption_reply(s)
            resp = send_prompt_to_lloyd(s["venue_context"], f"{s['prompt']}\n\n{follow}")

        txt = format_output(s, resp)
        out_path = save_txt(STAGING_DIR, n, txt)
        print(f"✅ Saved → {out_path}")
        produced_names.append(out_path.name)

        # move consumed input JSON → processed_inputs
        for cand in INPUT_DIR.glob(f"scenario_{n:03d}.json"):
            move_input_to_processed(cand)

    # Remember exactly what we just produced
    _save_manifest(LAST_RUN_MANIFEST, produced_names)
    return produced_names

def _evaluate(args) -> None:
    produced = run_evaluator(
        src_dir=Path(args.eval_src),
        dest_dir=Path(args.eval_dest),
        model_wlshd=args.eval_model,
        model_metrics=args.metrics_model or args.eval_model,
        inplace=args.eval_inplace,
        only_names=args.only_names,  # may be None; falls back to LAST_RUN_MANIFEST
    )
    if produced:
        print(f"🗂️  last_eval_outputs.json written with {len(produced)} file(s).")

def _inject(args, only_names: Optional[List[str]] = None) -> None:
    # If caller didn’t provide explicit list, use last eval batch
    only = only_names or args.only_names or _load_manifest(LAST_EVAL_MANIFEST) or None
    inject_to_kb(Path(args.inject_src), gate=float(args.gate), only_names=only)

def _rebuild(_args) -> None:
    rebuild_vectorstore()

# ---------- CLI ----------

def main() -> None:
    ensure_dirs()

    ap = argparse.ArgumentParser(
        description="Lloyd trainer + evaluator + metrics + gated KB injection (scoped)"
    )

    # Main stages
    ap.add_argument("--run", action="store_true", help="Run scenarios from scenario_inputs → scenario_outputs")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate .txt (fill WLSHD + write metrics JSON/CSV)")
    ap.add_argument("--inject", action="store_true", help="Inject evaluated files into KB if score_overall ≥ gate")
    ap.add_argument("--rebuild", action="store_true", help="After inject, rebuild vectorstore and tag vocab")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run → Evaluate → Inject → Rebuild in one shot (scoped to current run)",
    )

    # Scope list (exact basenames, e.g., scenario_091_output.txt or scenario_091_output_no01.txt)
    ap.add_argument(
        "--only-names",
        nargs="+",
        default=None,
        help="Limit evaluate/inject to these filenames (exact basenames).",
    )

    # Paths / options
    ap.add_argument("--eval-src", default=str(STAGING_DIR), help="Where to read .txt for evaluation (default: scenario_outputs)")
    ap.add_argument("--eval-dest", default=str(EVAL_DIR), help="Where to write evaluated .txt (default: evaluator_outputs)")
    ap.add_argument("--eval-inplace", action="store_true", help="Write WLSHD back into source files instead of dest folder")
    ap.add_argument("--inject-src", default=str(EVAL_DIR), help="Folder to read evaluated files from for KB injection")

    # Models
    ap.add_argument("--eval-model", default="gpt-4o", help="Model for WLSHD critique (default: gpt-4o)")
    ap.add_argument("--metrics-model", default=None, help="Model for metrics JSON (default: uses --eval-model if not set)")

    # Gate
    ap.add_argument("--gate", type=float, default=4.0, help="Minimum score_overall to inject into KB (default: 4.0)")

    args = ap.parse_args()

    # Guard: if running scenarios, ensure Lloyd is reachable
    if args.run or args.all:
        if not check_lloyd_health(LLOYD_API_URL):
            print("🚫 Aborting: Lloyd must be running (python app.py) before using --run/--all")
            return

    # One-shot pipeline
    if args.all:
        run_names = _run_scenarios()
        args.only_names = run_names
        _evaluate(args)
        _inject(args, run_names)
        _rebuild(args)
        return

    # Individual stages
    if args.run:
        _ = _run_scenarios()

    if args.evaluate:
        # If no explicit list from CLI, fall back to last run manifest
        if not args.only_names:
            args.only_names = _load_manifest(LAST_RUN_MANIFEST)
        _evaluate(args)

    if args.inject:
        _inject(args)  # uses args.only_names or LAST_EVAL_MANIFEST

    if args.rebuild:
        _rebuild(args)

if __name__ == "__main__":
    main()