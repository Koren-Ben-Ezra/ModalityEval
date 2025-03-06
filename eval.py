import pandas as pd

from benchmarkManager import BenchmarkManager
from filters import IdentityTextFilter, IdentityImageFilter, GaussianImageFilter, ContrastStretchingImageFilter, HistogramEqualizationImageFilter, TextBackgroundReplacementFilter, ScrambleLetterInWordsTextFilter, ScrambleWordsInSentenceTextFilter, PushFrontPhraseTextFilter
from datasetWrapper import GSM8kWrapper
from multimodalWrappers import ModelManager, LlamaWrapper

class Filters:
    """
    Filters is a singleton class that manages a collection of text and image filters.
    Attributes:
        img_filters (list): A list of image filter instances.
        text_filters (list): A list of text filter instances.
    """
    
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

def eval_Llama32vision_gsm8k():
    # prepare the benchmark
    metadata = {
        "Multimodal": "Llama-3.2-11B-Vision",
        "Dataset": "GSM8k"
    }
    model_manager = ModelManager(LlamaWrapper())
    datasetWrapper = GSM8kWrapper()
    
    # create the benchmark
    benchmark_manager = BenchmarkManager(datasetWrapper, model_manager, metadata, save_predictions=True)

    filters = Filters()
    
    # identity filters, just to check if the benchmark is working
    benchmark_manager.execute_test(filters.text_filters[0], filters.img_filters[0])
    
    benchmark_manager.write_summary()