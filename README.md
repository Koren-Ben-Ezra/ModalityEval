## Virtual Environment Setup

0. if miniconda 3 isn't already installed:
   https://www.anaconda.com/download/success

1. Create the virtural environment:

```
conda env create -f environment/environment.yml
- In case of errors, try using "pip3 install <package-name>" to install the problematic packages

```

2. Active the environment:

```
conda activate ModalityEval
```

3. (Optional) To re-install the requirements using pip 
(after conda environment was activated)
```
pip install -r environment\requirements.txt

```

4. tokens (model : token):
meta-llama/Llama-Guard-3-11B-Vision : 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
    

## Miro (view only)
https://miro.com/app/board/uXjVIWQC1F8=/