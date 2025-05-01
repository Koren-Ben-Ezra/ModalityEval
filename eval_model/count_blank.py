import os
import csv

OUTPUT_FN = "count_blank.csv"

def count_empty_last_column(folder='results'):
    """
    Read every CSV in `target_folder`, count how many rows have an empty value
    in the last column, and write a summary CSV named `count_blank.csv`
    with columns ['filename', 'blank_count'].
    """
    results = {}
    
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            lines = 0
            empty_count = 0
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    lines += 1
                    if row and row[-1].strip() == '':
                        empty_count += 1
            results[filename] = (empty_count, lines)

    return results

with open(OUTPUT_FN, "w", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["file", "empty_last_column_count"])

    results = count_empty_last_column()
    for file, (count, lines) in results.items():
        writer.writerow([file, count, round(count/lines, 2)])
