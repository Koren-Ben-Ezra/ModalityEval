import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import os
"""
    Load two summary CSVs (`csv1`, `csv2`), extract their noise probabilities `p`,
    prepend the p=0 manual row if missing, then plot both datasets (Shuffle vs. Flip2)
    on the same axes and save to PDF.
"""
# --- ACL proceedings style ---
mpl.rcParams['font.family']         = 'serif'
mpl.rcParams['font.serif']          = ['Times New Roman']
mpl.rcParams['text.usetex']         = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['axes.titlesize']      = 16
mpl.rcParams['axes.labelsize']      = 16
mpl.rcParams['xtick.labelsize']     = 14
mpl.rcParams['ytick.labelsize']     = 14
mpl.rcParams['legend.fontsize']     = 14
mpl.rcParams['lines.linewidth']     = 1.2
mpl.rcParams['lines.markersize']    = 4
mpl.rcParams['grid.linestyle']      = '--'
mpl.rcParams['grid.linewidth']     = 0.5

# Prompt user for CSV paths
csv1 = input("Enter path to first CSV file (e.g., summary_shuffle_separate.csv): ").strip()
csv2 = input("Enter path to second CSV file (e.g., summary_flip2_separate.csv): ").strip()

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

df1['p'] = df1['TF_filter'].apply(
    lambda f: float(re.search(r'_(\d+\.\d+)_', f).group(1))
)
df2['p'] = df2['TF_filter'].apply(
    lambda f: float(re.search(r'_(\d+\.\d+)_', f).group(1))
)

if not any(df1['p'] == 0.0):
    df1 = pd.concat([
        pd.DataFrame({
            'p': [0.0],
            'TF_filter': [f'{os.path.splitext(os.path.basename(csv1))[0].split("summary_")[1]}_0.00_TF.csv'],
            'IF_filter': [f'{os.path.splitext(os.path.basename(csv1))[0].split("summary_")[1]}_0.00_IF.csv'],
            'TF_accuracy': [0.82],
            'IF_accuracy': [0.55]
        }),
        df1
    ], ignore_index=True)
if not any(df2['p'] == 0.0):
    df2 = pd.concat([
        pd.DataFrame({
            'p': [0.0],
            'TF_filter': [f'{os.path.splitext(os.path.basename(csv2))[0].split("summary_")[1]}_0.00_TF.csv'],
            'IF_filter': [f'{os.path.splitext(os.path.basename(csv2))[0].split("summary_")[1]}_0.00_IF.csv'],
            'TF_accuracy': [0.82],
            'IF_accuracy': [0.55]
        }),
        df2
    ], ignore_index=True)

df1 = df1.sort_values('p')
df2 = df2.sort_values('p')

name1 = re.search(r'summary_(.*)_separate', os.path.basename(csv1)).group(1).title()
name2 = re.search(r'summary_(.*)_separate', os.path.basename(csv2)).group(1).title()

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(df1['p'], df1['TF_accuracy'], marker='o', linestyle='-', label=f'Txt Acc ({name1})')
ax.plot(df1['p'], df1['IF_accuracy'], marker='s', linestyle='-', label=f'Img Acc ({name1})')
ax.plot(df2['p'], df2['TF_accuracy'], marker='^', linestyle='--', label=f'Txt Acc ({name2})')
ax.plot(df2['p'], df2['IF_accuracy'], marker='v', linestyle='--', label=f'Img Acc ({name2})')

ax.set_xlabel(r'Noise Probability $p$')
ax.set_ylabel('Accuracy')
ax.set_title(f'Text vs. Image Accuracy for {name1} and {name2} Perturbations')

ax.grid(True)
ax.legend(frameon=False)

output_file = f'fig_{name1.lower()}_{name2.lower()}_accuracy_acl.pdf'
fig.tight_layout()
fig.savefig(output_file, format='pdf')
print(f"Plot saved to {output_file}")

plt.show()
