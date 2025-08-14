import os

folder_path = "system_prompt_mods"  # adjust if your folder is named differently

for filename in os.listdir(folder_path):
    if not filename.endswith(".txt"):
        continue  # skip non-text files
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Only replace if 'Tags:' exists and 'Keywords:' doesn't
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith("Tags:") and "Keywords:" not in line:
            lines[i] = line.replace("Tags:", "Keywords:", 1)
            updated = True
            break

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Updated: {filename}")
    else:
        print(f"No change: {filename}")