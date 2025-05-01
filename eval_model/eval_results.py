import os
import csv
from decimal import Decimal, InvalidOperation
"""
    Scan all raw result CSVs in `input_folder`, extract key metrics
    (e.g. filter name, correct/total, accuracy), and write them into
    `output_csv` with one row per filter.
"""

def clean_str_number(s: str) -> str:
        if not s:
            return s
        
        try:
            d = Decimal(s)
        except InvalidOperation:
            return s
        
        d_normalized = d.normalize()
        s_formatted = format(d_normalized, 'f')
        
        if '.' in s_formatted:
            s_formatted = s_formatted.rstrip('0').rstrip('.')
        
        return s_formatted

DIR_NAME = input("Enter the directory name (e.g., 'shuffle'): ")
TRIM_DIR = input("Enter the directory name for trimmed results (default: 'results_trimmed_'): ") or "results_trimmed_" + DIR_NAME
INPUT_DIR = "./new_results/" + DIR_NAME
OUTPUT_FN  = "summary_" + DIR_NAME + ".csv"   
os.makedirs(TRIM_DIR, exist_ok=True)

with open(OUTPUT_FN, "w", newline="") as out_f:
    writer = csv.writer(out_f)
    # header
    writer.writerow(["filter", "correct", "total", "accuracy"])
    
    for fn in sorted(os.listdir(INPUT_DIR)):
        if not fn.lower().endswith(".csv"):
            continue
        in_path   = os.path.join(INPUT_DIR, fn)
        trim_path = os.path.join(
            TRIM_DIR,
            fn.rsplit(".csv", 1)[0] + "_trimmed.csv"
        )

        total, matches = 0, 0
        trimmed_rows   = []

        with open(in_path, newline="") as in_f:
            reader = csv.reader(in_f)
            for row in reader:
                if not row:
                    continue
                if len(row) == 4:
                    row = row[:3]
                row[2] = clean_str_number(row[2])
                if row[2] == "00" or row[2] == "000" or row[2] == "0000":
                    row[2] = "0"

                trimmed_rows.append(row)

                total += 1
                if row[1] == row[2]:
                    matches += 1
        with open(trim_path, "w", newline="") as tf:
            writer_t = csv.writer(tf)
            writer_t.writerows(trimmed_rows)

        frac = matches/total if total else 0
        writer.writerow([fn, matches, total, f"{frac:.2f}"])
        
print(f"Done!  Summaries written to {OUTPUT_FN}")
