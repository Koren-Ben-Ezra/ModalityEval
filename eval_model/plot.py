import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import os
"""
    Load a single summary CSV from `csv_path`, extract the probability `p`
    from the 'TF_filter' filenames, sort by `p`, and plot TF vs. IF accuracy
    as a function of `p`, saving to PDF.
"""
# --- ACL proceedings style ---
mpl.rcParams['font.family']         = 'serif'
mpl.rcParams['font.serif']          = ['Times New Roman']
mpl.rcParams['text.usetex']         = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['axes.titlesize']      = 16   # 10pt for ACL axis titles
mpl.rcParams['axes.labelsize']      = 16   # 10pt for axis labels
mpl.rcParams['xtick.labelsize']     = 14   # 8pt for tick labels
mpl.rcParams['ytick.labelsize']     = 14
mpl.rcParams['legend.fontsize']     = 14    # 8pt for legend text
mpl.rcParams['lines.linewidth']     = 1.2  # slightly thicker lines
mpl.rcParams['lines.markersize']    = 4    # modest marker size
mpl.rcParams['grid.linestyle']      = '--'
mpl.rcParams['grid.linewidth']      = 0.5

csv_path = input("Enter path to CSV file: ").strip()

df = pd.read_csv(csv_path)

df['p'] = df['TF_filter'].apply(
    lambda f: float(re.search(r'_(\d+\.\d+)_', f).group(1))
)

df = df.sort_values('p')
print(df)
fig, ax = plt.subplots(figsize=(6.4, 3.6))  # matches ACL column width

ax.plot(df['p'], df['TF_accuracy'],
        marker='o', color='red',   label='Text Accuracy')
ax.plot(df['p'], df['IF_accuracy'],
        marker='s', color='blue',  label='Image Accuracy')

ax.set_xlabel(r'Shuffle Probability $p$')
ax.set_ylabel('Accuracy')

base = os.path.basename(csv_path)
title_str = os.path.splitext(base)[0].replace('_', ' ').title()
ax.set_title(title_str)

ax.grid(True)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig('fig_text_image_accuracy_acl.pdf', format='pdf')

plt.show()
