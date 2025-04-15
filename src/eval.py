import json
from PIL import Image
import multiprocessing
import torch
from queue import Queue

from src.text2image import FixedSizeText2Image
from src.benchmarkManager import BenchmarkManager
from src.filters import *
from src.datasetWrapper import GSM8kWrapper
from src.multimodalWrappers import LlamaWrapper
from src.log import Log
PARAMETERS_PATH = "parameters.json"



def eval_llama():
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
    ## Without slurm:
    # text2image = FixedSizeText2Image()
    
    datasetWrapper = GSM8kWrapper(text2image)
    
    metadata = {
        "test name": "basic test",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
        "Save Predictions": True,
    }
    # Read the JSON file
    with open(PARAMETERS_PATH, 'r') as file:
        data = json.load(file)
        saved_text = data.get('saved text', '')


    # Initialize the BenchmarkManager
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark for {multimodal_wrapper.model_name} on {datasetWrapper.dataset_id}")
    
    # ------ execute text and image filter tests ------ #
    # -- Identity filters -- #
    # Text #
    # benchmark_manager.execute_test(IdentityTextFilter()) DONE!
    # Image #
    benchmark_manager.add_job(IdentityImageFilter())
    
    
    # -- Noise filters -- #
    # Text #
    benchmark_manager.add_job(ShuffleWordTextFilter()) #p = 0.2
    benchmark_manager.add_job(SwapWordsTextFilter()) #p = 0.2
    # Image #
    benchmark_manager.add_job(img_f=HistogramEqualizationImageFilter())
    benchmark_manager.add_job(img_f=GaussianImageFilter()) # kernel_size = 5 sigms = 1.0


    # -- General information filters -- #
    # Text #
    phrase = saved_text.get("scared", "")
    benchmark_manager.add_job(PushFrontTextFilter(phrase))
    
    # Image #
    benchmark_manager.add_job(img_f=ReplaceBackgroundImageFilter()) # alpha: float=0.5
    image_path = f"images/amanda.jpg"
    image = Image.open(image_path)
    benchmark_manager.add_job(img_f=PushTopImageFilter(image)) # additional_image: Image
    benchmark_manager.add_job(img_f=ReplaceBackgroundImageFilter(image)) #  background_image: Image , alpha: float=0.5
    
    
    # -- Personal information filters -- #
    # Text #
    benchmark_manager.add_job(SurroundByCorrectAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6
    benchmark_manager.add_job(SurroundByWrongAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6)
    benchmark_manager.add_job(SurroundByPartialCorrectAnsTextFilter()) #  p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 6
    # Image #
    benchmark_manager.add_job(img_f=SurroundByCorrectAnsImageFilter()) # num_repeats: int = 5, alpha: float = 0.2,  font_size: int = 40, font_type: str = "arial.ttf", font_color = "black"
    benchmark_manager.add_job(img_f=SurroundByWrongAnsImageFilter()) # same
    benchmark_manager.add_job(img_f=SurroundByPartialCorrectAnsImageFilter()) # p = 0.2, the rest as SurroundByCorrectAnsImageFilter
    
    # --------------------------------------- #

    execute_jobs(benchmark_manager.category_queue)
    Log().logger.info("Evaluation completed.")
    
    
    ##########################################################################