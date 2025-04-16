import json
import os

from src.text2image import FixedSizeText2Image, FilteredFixedSizeText2Image
from src.benchmarkManager import BenchmarkManager
from src.filters import *
from src.datasetWrapper import GSM8kWrapper
from src.multimodalWrappers import LlamaWrapper
from src.log import Log
PARAMETERS_PATH = "parameters.json"

job = os.getenv("JOB", "0")
# JOB 1: Identity filters
# JOB 2: Noise filters
# JOB 3: Noise filters
# JOB 4: General information filters
# JOB 5: General information filters
# JOB 6: Personal information filters
# JOB 7: Personal information filters
# JOB B: Shuffle word test

if job == "0":
    raise ValueError("Please set the JOB environment variable (sbatch run_slurm.sh <job>)")

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
        parameters = json.load(file)
    
    try:
        saved_text = parameters['saved text']
    except KeyError:
        Log().logger.error("The key 'saved text' was not found in the JSON file.")
        return

    # Initialize the BenchmarkManager
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark for {multimodal_wrapper.model_name} on {datasetWrapper.dataset_id}")
    
    # ------ execute text and image filter tests ------ #
    # -- Identity filters -- #
    if job == "1":
        # Text #
        benchmark_manager.register_job(text_f=IdentityTextFilter())
        # Image #
        benchmark_manager.register_job(text_f=IdentityImageFilter())
    
    
    # -- Noise filters -- #
    if job == "2":
        # TODO: fix those two filters:
        # Text #
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter()) #p = 0.2
        benchmark_manager.register_job(SwapWordsTextFilter()) #p = 0.2
    if job == "3":
        # Image #
        benchmark_manager.register_job(img_f=HistogramEqualizationImageFilter())
        benchmark_manager.register_job(img_f=GaussianImageFilter()) # kernel_size = 5 sigms = 1.0


    # -- General information filters -- #
    if job == "4":
        # Text #
        try:
            phrase_scared = saved_text["scared"]
            benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase_scared))
        except KeyError:
            Log().logger.error("The key 'scared' was not found in the JSON file.")
        
        try:                
            phrase_sad = saved_text["sad"]
            benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase_sad))
        except KeyError:
            Log().logger.error("The key 'sad' was not found in the JSON file.")
    
    
    if job == "5":
        # Image #
        image_path = f"images/amanda.jpg"
        image = Image.open(image_path)
        benchmark_manager.register_job(img_f=ReplaceBackgroundImageFilter(image)) # alpha: float=0.5
        benchmark_manager.register_job(img_f=PushTopImageFilter(image)) # additional_image: Image
        benchmark_manager.register_job(img_f=ReplaceBackgroundImageFilter(image)) #  background_image: Image , alpha: float=0.5
    
    
    # -- Personal information filters -- #
    if job == "6":
        # Text #
        benchmark_manager.register_job(text_f=SurroundByCorrectAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6
        benchmark_manager.register_job(text_f=SurroundByWrongAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6)
        benchmark_manager.register_job(text_f=SurroundByPartialCorrectAnsTextFilter()) #  p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 6
    
    if job == "7":
        # Image #
        benchmark_manager.register_job(img_f=SurroundByCorrectAnsImageFilter()) # num_repeats: int = 5, alpha: float = 0.2,  font_size: int = 40, font_type: str = "arial.ttf", font_color = "black"
        benchmark_manager.register_job(img_f=SurroundByWrongAnsImageFilter()) # same
        benchmark_manager.register_job(img_f=SurroundByPartialCorrectAnsImageFilter()) # p = 0.2, the rest as SurroundByCorrectAnsImageFilter
    
    # --------------------------------------- #

    benchmark_manager.start_workers()
    
    ##########################################################################
    

def eval_llama2():
    
    if job != "B":
        return
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    text_filter = ShuffleWordTextFilter()
    text2image=FilteredFixedSizeText2Image(text_filter, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
    ## Without slurm:
    # text2image = FixedSizeText2Image()
    
    datasetWrapper = GSM8kWrapper(text2image, cache_filename="gsm8k_shuffled_dataset")
    
    metadata = {
        "test name": "shuffle word test",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
        "Save Predictions": True,
    }

    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark for {multimodal_wrapper.model_name} on {datasetWrapper.dataset_id}")
    
    benchmark_manager.register_job(text_f=ShuffleWordTextFilter(), img_f=IdentityImageFilter())
    
    benchmark_manager.start_workers()
    
    
# def eval_llama3():
    
#     if job != "C":
#         return
    
#     # prepare the benchmark
#     Log().logger.info("------------------------------------------------")
#     Log().logger.info("Starting evaluation...")
    
#     multimodal_wrapper = LlamaWrapper()
    
#     ############################# GSM8k dataset ##############################
    
#     ## With slurm:
#     text2image=FixedSizeText2Image(font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
#     ## Without slurm:
#     # text2image = FixedSizeText2Image()
    
#     datasetWrapper = GSM8kWrapper(text2image, cache_filename="gsm8k_shuffled_dataset")
    
#     metadata = {
#         "test name": "shuffle word test",
#         "Model": multimodal_wrapper.model_name,
#         "Dataset": datasetWrapper.dataset_id,
#         "Save Predictions": True,
#     }