import os
import json
import requests
from datetime import datetime
from pathlib import Path
import subprocess
import argparse

# ========== Config ==========
INPUT_DIR = Path("training_loop/scenario_inputs")
OUTPUT_DIR = Path("training_loop/scenario_outputs")
KB_DIR = Path("knowledge_base/training_modules/scenario_runs")
LLOYD_API_URL = "http://localhost:5000/"
TAG_SCRIPT_PATH = Path("scripts/update_tag_vocab.py")

# Parse CLI args
parser = argparse.ArgumentParser(description="Run Lloyd scenario trainer.")
parser.add_argument("--inject", action="store_true", help="Inject outputs into KB and rebuild vectorstore.")
args = parser.parse_args()

# Ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)

# ========== Core Functions ==========

def load_scenarios():
    scenarios = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            path = INPUT_DIR / filename
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    scenarios.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to load {filename}: {e}")
    return scenarios

def send_prompt_to_lloyd(venue, user_prompt):
    payload = {
        "venue_concept": venue,
        "user_prompt": user_prompt,
        "use_live_search": False
    }
    try:
        response = requests.post(LLOYD_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("assistant_response", "[ERROR] No assistant response found.")
    except Exception as e:
        return f"[ERROR] Failed to reach Lloyd: {e}"

def format_output(scenario, lloyd_response):
    output = []
    output.append(f"Title: {scenario['title']}")
    output.append(f"Tags: {', '.join(scenario['tags'])}")
    output.append("Scenario Type: Training")
    output.append(f"System Mod: {scenario['system_mod']}")
    output.append(f"Venue Context: {scenario['venue_context']}")
    output.append(f"Prompt: {scenario['prompt']}")
    output.append("Lloyd's Response:")
    output.append(lloyd_response)
    output.append("\nWhat Lloyd Should Have Done:\n[PLACEHOLDER]\n")
    output.append(f"(Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    return "\n\n".join(output)

def save_output_file(scenario_number, formatted_text):
    filename = f"scenario_{scenario_number}_output.txt"
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"✅ Saved to: {filepath}")

def inject_into_kb(scenario_number, formatted_text):
    filename = f"scenario_{scenario_number}_output.txt"
    filepath = KB_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"📚 Injected into KB: {filepath}")

def rebuild_vectorstore():
    print("🔁 Rebuilding vectorstore...")
    subprocess.run(["python", "kb_loader.py", "--rebuild"], check=True)

    if TAG_SCRIPT_PATH.exists():
        print("🏷️ Updating tag vocab...")
        subprocess.run(["python", str(TAG_SCRIPT_PATH)], check=True)
    else:
        print("⚠️ Tag vocab script not found. Skipping.")

# ========== Main ==========

def run_trainer():
    print("🔁 Loading scenarios...")
    scenarios = load_scenarios()
    if not scenarios:
        print("⚠️ No scenarios found.")
        return

    for scenario in scenarios:
        print(f"\n🚀 Running Scenario {scenario['scenario_number']}: {scenario['title']}")
        lloyd_response = send_prompt_to_lloyd(
            venue=scenario["venue_context"],
            user_prompt=scenario["prompt"]
        )
        formatted = format_output(scenario, lloyd_response)

        if args.inject:
            inject_into_kb(scenario["scenario_number"], formatted)
        else:
            save_output_file(scenario["scenario_number"], formatted)

    if args.inject:
        rebuild_vectorstore()

if __name__ == "__main__":
    run_trainer()