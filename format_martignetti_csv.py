import pandas as pd
import os

print("Working directory:", os.getcwd())

csv_path = "./knowledge_base/portfolios/martignetti.csv"
output_path = "./knowledge_base/portfolios/martignetti_formatted.txt"

def is_valid_string(s):
    return isinstance(s, str) and s.strip() != "" and s.strip().lower() != "nan"

def generate_product_blurbs(csv_path, output_path):
    try:
        df = pd.read_csv(csv_path)
        blurbs = []

        for _, row in df.iterrows():
            brand = row.get("Brand", "")
            name = row.get("Product Name", "")
            if not is_valid_string(brand) or not is_valid_string(name):
                continue  # Skip incomplete rows

            category = row.get("Category", "unspecified category")
            abv = row.get("ABV", "N/A")
            price = row.get("Price (Per Case)", "unknown price")
            size = row.get("Size", "unknown size")
            pack = row.get("Case Pack", "unknown quantity")
            region = row.get("Region", "unknown region")
            distributor = row.get("Distributor", "Martignetti")

            blurb = (
                f"{distributor} distributes {brand} {name} — a {category} "
                f"({abv} ABV). Priced at {price} per case ({pack} x {size}). "
                f"Available in {region}."
            )
            blurbs.append(blurb)

        if not blurbs:
            print("⚠️ No valid product blurbs generated. Check your CSV content.")
        else:
            with open(output_path, "w") as f:
                for blurb in blurbs:
                    f.write(f"{blurb}\n")
            print(f"✅ Wrote {len(blurbs)} product blurbs to {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_product_blurbs(csv_path, output_path)