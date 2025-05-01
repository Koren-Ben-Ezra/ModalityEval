## Virtual Environment Setup

0. if miniconda 3 isn't already installed:
   https://www.anaconda.com/download/success

1. Create the virtural environment:

```
conda env create -f environment/environment.yml
```
- In case of errors, try using "pip install <package-name>" to install the problematic packages


2. Active the environment:

```
conda activate ModalityEval
```

3. (Optional) To re-install the requirements using pip 
(after conda environment was activated)
```
pip install -r environment\requirements.txt
pip install --upgrade transformer

```

4. login token for hugging face:
```
huggingface-cli login
```
5. Token: 
```
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
```

## Miro (view only)
https://miro.com/app/board/uXjVIWQC1F8=/

## Evaluation
All evaluation code resides in eval_model/. Below is how to use each script:
1. eval_results.py Aggregates per-file evaluation metrics into a summary CSV (e.g., eval_summary.csv). Reads raw result .csv files in eval_model/results/ and writes a consolidated CSV.
2. separate_csv.py Splits a summary CSV into separate TF/IF accuracy columns. Reads summary_<name>.csv and writes seperate/summary_<name>_separate.csv.
3. count_blank.py Counts empty entries in the last column of every .csv in a target folder. Outputs count_blank.csv in the current directory.
4. plot.py Generates a plot (fig_text_image_accuracy_acl.pdf) of accuracy vs. shuffle probability for text and image inputs.
5. plot2csv.py Creates a combined plot of 2 csv files.
