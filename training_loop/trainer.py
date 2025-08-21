import os
import json
import requests
from datetime import datetime

# Paths
INPUT_DIR = "training_loop/scenario_inputs"
OUTPUT_DIR = "training_loop/scenario_outputs"
LLOYD_API_URL = "http://localhost:5000/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_scenarios():
    """Load all JSON scenario files from the input directory."""
    scenarios = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            path = os.path.join(INPUT_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    scenarios.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to load {filename}: {e}")
    return scenarios

def send_prompt_to_lloyd(venue, user_prompt):
    """Send prompt to Lloyd's API and return the assistant's response."""
    payload = {
        "venue_concept": venue,
        "user_prompt": user_prompt,
        "use_live_search": False
    }
    try:
        response = requests.post(LLOYD_API_URL, json=payload)
        response.raise_for_status()
        try:
            return response.json().get("assistant_response", "[ERROR] No assistant response found.")
        except Exception as e:
            return f"[ERROR] Could not parse response JSON: {e}"
    except Exception as e:
        return f"[ERROR] Failed to reach Lloyd: {e}"

def format_output(scenario, lloyd_response):
    """Format the scenario run output into .txt format."""
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
    """Save the scenario output to a .txt file."""
    filename = f"scenario_{scenario_number}_output.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"✅ Saved: {filepath}")

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
        save_output_file(scenario["scenario_number"], formatted)

if __name__ == "__main__":
    run_trainer()