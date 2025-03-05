from benchmarkManager import BenchmarkManager
from filters import IdentityTextFilter, IdentityImageFilter, GaussianImageFilter, ContrastStretchingImageFilter, HistogramEqualizationImageFilter, TextBackgroundReplacementFilter, ScrambleLetterInWordsTextFilter, ScrambleWordsInSentenceTextFilter, PushFrontPhraseTextFilter
from datasetWrapper import GSM8kWrapper
from multimodalWrappers import ModelManager, Llama32_11B_visionWrapper
import pandas as pd

class Filters:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Filters, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):    
        self.img_filters = []
        self.text_filters = []
    
        # Text filters
        self.text_filters.append(IdentityTextFilter())
        self.text_filters.append(TextBackgroundReplacementFilter())
        self.text_filters.append(ScrambleLetterInWordsTextFilter())
        self.text_filters.append(ScrambleWordsInSentenceTextFilter())
        # self.text_filters.append(PushFrontPhraseTextFilter(phrase))
        
        # Image filters
        self.img_filters.append(IdentityImageFilter())
        self.img_filters.append(GaussianImageFilter())
        self.img_filters.append(ContrastStretchingImageFilter())
        self.img_filters.append(HistogramEqualizationImageFilter())


def write_csv_file(filename: str, df: pd.DataFrame, metadata: dict):
# Save to CSV with metadata as a header
    with open(filename, "w") as file:
        for key, value in metadata.items():
            file.write(f"# {key}: {value}\n")  # Writing metadata as comments
        df.to_csv(file, index=False)


def eval_Llama32vision_gsm8k():
    metadata = {
        "Multimodal": "Llama-3.2-11B-Vision",
        "Dataset": "GSM8k"
    }
    
    filters = Filters()
    model_manager = Llama32_11B_visionWrapper()
    benchmark_manager = BenchmarkManager(GSM8kWrapper(), model_manager)
        
    df = pd.DataFrame(columns=["Text Filter", "Image Filter", "Text Accuracy", "Image Accuracy"])
    
    # identity filters, just to check if the benchmark is working
    text_accuracy, img_accuracy = benchmark_manager.execute_test(filters.text_filters[0], filters.img_filters[0])
    df = df.append({"Text Filter": "IdentityTextFilter", "Image Filter": "IdentityImageFilter", "Text Accuracy": text_accuracy, "Image Accuracy": img_accuracy}, ignore_index=True)
    
    write_csv_file("results.csv", df, metadata)
