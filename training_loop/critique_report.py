import re
from pathlib import Path
from collections import Counter
import pandas as pd

# ---------- Paths ----------
CRITIQUE_DIR = Path("training_loop/critique_only")
REPORT_CSV = Path("analytics/critique_report.csv")

# ---------- Keywords to track ----------
KEYWORDS = {
    "contingency": ["contingency", "backup", "redundancy", "plan b"],
    "throughput": ["throughput", "drinks/minute", "service speed", "tickets/hour"],
    "training": ["training", "briefing", "checklist", "onboarding", "staff module"],
    "waste": ["waste", "compost", "recycle", "spoilage"],
    "batching": ["batching", "pre-batch", "batch size", "shelf life"],
    "menu_alignment": ["menu cohesion", "theme", "brand", "guest profile"],
    "costing": ["cost", "pour cost", "margin", "profit"],
    "kpi": ["KPI", "metric", "scorecard", "review"],
}

def extract_wlshd(text: str) -> str:
    """Grab only the WLSHD critique section from the file."""
    m = re.search(r"What Lloyd Should Have Done:(.*?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def scan_critiques():
    counts = Counter()
    details = []

    for p in sorted(CRITIQUE_DIR.glob("scenario_*_output.txt")):
        text = p.read_text(encoding="utf-8")
        wlshd = extract_wlshd(text)
        if not wlshd:
            continue

        # keyword tally
        for label, patterns in KEYWORDS.items():
            if any(re.search(rf"\b{pat}\b", wlshd, re.IGNORECASE) for pat in patterns):
                counts[label] += 1

        details.append({"file": p.name, "wlshd": wlshd})

    return counts, details

def main():
    CRITIQUE_DIR.mkdir(parents=True, exist_ok=True)
    counts, details = scan_critiques()

    total = sum(counts.values())
    print("📊 Critique Frequency Report")
    print("="*40)
    for k, v in counts.most_common():
        pct = (v/total*100) if total else 0
        print(f"{k:15s} {v:3d}  ({pct:.1f}%)")

    # Export CSV of detailed WLSHDs for deeper analysis
    pd.DataFrame(details).to_csv(REPORT_CSV, index=False)
    print(f"\n💾 Detailed WLSHDs exported to {REPORT_CSV}")

if __name__ == "__main__":
    main()