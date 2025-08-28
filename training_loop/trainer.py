# training_loop/trainer.py
# ------------------------------------------------------------
# Lloyd Training Loop v3 (integrated evaluator + metrics gate)
# - Run scenarios -> Lloyd output (.txt)
# - Evaluate -> fill WLSHD + write metrics JSON + rolling CSV
# - Gate -> inject to KB if score_overall >= gate, else send to critique_only
# - Auto organizes working folders (processed_inputs/ etc.)
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

import requests
from dotenv import load_dotenv

load_dotenv()  # .env at project root

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
    path = directory / _scenario_out_name(scenario_number)
    path.write_text(text, encoding="utf-8")
    return path


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
    # Trim trailing whitespace in values
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
        # Validate keys; fill missing with defaults
        for k in ["score_overall", "clarity", "feasibility", "structure", "actionability", "alignment", "notes"]:
            if k not in data:
                data[k] = base[k]
        data["scored_at"] = base["scored_at"]
        return data
    except Exception as e:
        base["error"] = f"OpenAI metrics failed: {e}"
        return base


# ---------- Evaluation runners ----------

def run_evaluator(src_dir: Path, dest_dir: Path, model_wlshd: str, model_metrics: str, inplace: bool = False) -> None:
    files = sorted(src_dir.glob("scenario_*_output.txt"))
    print(f"🔎 Evaluator scanning: {src_dir}  |  {len(files)} file(s)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not files:
        print("⚠️  No .txt files found to evaluate (run --run first).")
        return

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

        # 2) Write evaluated file
        if inplace:
            p.write_text(content, encoding="utf-8")
            out_path = p
            print(f"💬 Evaluated (in-place): {p.name}")
        else:
            out_path = dest_dir / p.name
            out_path.write_text(content, encoding="utf-8")
            print(f"💬 Evaluated → {out_path}")

        # 3) Always create metrics JSON (even if later gated)
        metrics = evaluate_metrics(content, model=model_metrics)
        # Master metrics copy under evaluator_outputs/metrics
        scenario_num = _extract_scenario_number_from_name(p.name)
        m_master = EVAL_METRICS_DIR / _scenario_metrics_name(scenario_num) if scenario_num is not None else (
            EVAL_METRICS_DIR / (p.stem + ".metrics.json")
        )
        m_master.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # Sibling copy (so other tools can find it)
        m_sibling = out_path.with_suffix(".metrics.json")
        m_sibling.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # 4) Append to rolling CSV
        _append_metrics_csv(m_master, metrics, source_txt=out_path)


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
    """
    Locate metrics for evaluated text `p` and return score_overall (float),
    or None if not found / invalid.
    Tries sibling first, then EVAL_METRICS_DIR using both naming styles.
    """
    # 1) sibling metrics
    sib = p.with_suffix(".metrics.json")
    if sib.exists():
        try:
            data = json.loads(sib.read_text(encoding="utf-8"))
            return float(data.get("score_overall"))
        except Exception:
            pass

    # 2) named by scenario number in metrics dir
    num = _extract_scenario_number_from_name(p.name)
    if num is not None:
        cand = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                return float(data.get("score_overall"))
            except Exception:
                pass

    # 3) same stem with .metrics.json in metrics dir
    cand2 = EVAL_METRICS_DIR / _sibling_metrics_name(p.name)
    if cand2.exists():
        try:
            data = json.loads(cand2.read_text(encoding="utf-8"))
            return float(data.get("score_overall"))
        except Exception:
            pass

    return None


def _copy_metrics_for(p: Path, dest_dir: Path) -> None:
    """
    Copy whichever metrics file exists for text p into dest_dir
    (sibling first, else metrics dir variants).
    """
    # sibling
    sib = p.with_suffix(".metrics.json")
    if sib.exists():
        (dest_dir / sib.name).write_text(sib.read_text(encoding="utf-8"), encoding="utf-8")
        return

    # metrics dir by number
    num = _extract_scenario_number_from_name(p.name)
    if num is not None:
        cand = EVAL_METRICS_DIR / _scenario_metrics_name(num)
        if cand.exists():
            (dest_dir / cand.name).write_text(cand.read_text(encoding="utf-8"), encoding="utf-8")
            return

    # metrics dir sibling name
    cand2 = EVAL_METRICS_DIR / _sibling_metrics_name(p.name)
    if cand2.exists():
        (dest_dir / cand2.name).write_text(cand2.read_text(encoding="utf-8"), encoding="utf-8")
        return


def inject_to_kb(from_dir: Path, gate: float) -> None:
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
            dest = CRITIQUE_DIR / p.name
            dest.write_text(txt, encoding="utf-8")
            _copy_metrics_for(p, CRITIQUE_DIR)
            diverted += 1
            continue

        if score >= gate:
            # 1) Inject to KB
            kb_target = KB_DIR / p.name
            kb_target.write_text(txt, encoding="utf-8")
            _copy_metrics_for(p, KB_DIR)
            # 2) Place a copy in injected_outputs for audit
            inj_target = INJECTED_DIR / p.name
            inj_target.write_text(txt, encoding="utf-8")
            _copy_metrics_for(p, INJECTED_DIR)
            print(f"📚 Injected (score={score:.2f}) → KB & injected_outputs: {p.name}")
            injected += 1
        else:
            dest = CRITIQUE_DIR / p.name
            dest.write_text(txt, encoding="utf-8")
            _copy_metrics_for(p, CRITIQUE_DIR)
            print(f"🧯 Diverted (score={score:.2f} < gate {gate:.2f}) → critique_only: {p.name}")
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


def _evaluate(args) -> None:
    run_evaluator(
        src_dir=Path(args.eval_src),
        dest_dir=Path(args.eval_dest),
        model_wlshd=args.eval_model,
        model_metrics=args.metrics_model or args.eval_model,
        inplace=args.eval_inplace,
    )


def _inject(args) -> None:
    inject_to_kb(Path(args.inject_src), gate=float(args.gate))


def _rebuild(_args) -> None:
    rebuild_vectorstore()


# ---------- CLI ----------

def main() -> None:
    ensure_dirs()

    ap = argparse.ArgumentParser(
        description="Lloyd trainer + evaluator + metrics + gated KB injection"
    )

    # Main stages
    ap.add_argument("--run", action="store_true", help="Run scenarios from scenario_inputs → scenario_outputs")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate .txt (fill WLSHD + write metrics JSON/CSV)")
    ap.add_argument("--inject", action="store_true", help="Inject evaluated files into KB if score_overall ≥ gate")
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
    if args.all:
        args.run = True
        args.evaluate = True
        args.inject = True
        args.rebuild = True
        args.eval_src = str(STAGING_DIR)
        args.eval_dest = str(EVAL_DIR)
        args.inject_src = str(EVAL_DIR)

    if args.run:
        _run_scenarios()

    if args.evaluate:
        _evaluate(args)

    if args.inject:
        _inject(args)

    if args.rebuild:
        _rebuild(args)


if __name__ == "__main__":
    main()