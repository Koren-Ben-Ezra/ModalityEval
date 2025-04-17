import json
import os

from src.text2image import FixedSizeText2Image, FilteredFixedSizeText2Image
from src.benchmarkManager import BenchmarkManager
from src.filters import *
from src.datasetWrapper import GSM8kWrapper
from src.multimodalWrappers import LlamaWrapper
from src.log import Log

PARAMETERS_PATH = "parameters.json"
slurm_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

selected_eval = os.getenv("SELECTED_EVAL", "0")
selected_task = os.getenv("SELECTED_TASK", "0")

# EVAL A: basic_eval_all
    # JOB 1: Identity filters
    # JOB 2: Noise filters
    # JOB 3: Noise filters
    # JOB 4: General information filters 
    # JOB 5: General information filters
    # JOB 6: General information filters 
    # JOB 7: Personal information filters
    # JOB 8: Personal information filters

# EVAL B: shuffle_txt_in_img_eval

# EVAL C: shuffle_p_increase_eval
    # JOB 1: shuffle word test p=0.05, 0.1, 0.15
    # JOB 2: shuffle word test p=0.25, 0.3, 0.35


image_path_amanda = os.path.join("images", "amanda.jpg")
image_path_angry = os.path.join("images", "angry.jpg")
image_path_relax = os.path.join("images", "relax.jpg")

image_amanda = Image.open(image_path_amanda)
image_angry = Image.open(image_path_angry)
image_relax = Image.open(image_path_relax)


if selected_eval == "0" or selected_task == "0":
    raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <task>'")

def basic_eval_all():
    if selected_eval != "A":
        return
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
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
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    # ------ execute text and image filter tests ------ #
    # -- Identity filters -- #
    if selected_task == "1":
        # Text #
        benchmark_manager.register_job(text_f=IdentityTextFilter())
        # Image #
        benchmark_manager.register_job(text_f=IdentityImageFilter())
    
    
    # -- Noise filters -- #
    if selected_task == "2":
        # Text #
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter()) #p = 0.2
        benchmark_manager.register_job(SwapWordsTextFilter()) #p = 0.2
    if selected_task == "3":
        # Image #
        benchmark_manager.register_job(img_f=HistogramEqualizationImageFilter())
        benchmark_manager.register_job(img_f=GaussianImageFilter()) # kernel_size = 5 sigms = 1.0


    # -- General information filters -- #
    if selected_task == "4":
        # Text #
        try:
            phrase_scared = saved_text["scared"]
            benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase_scared), inner_dir="scared_text")
        except KeyError:
            Log().logger.error("The key 'scared' was not found in the JSON file.")
        
        try:                
            phrase_sad = saved_text["sad"]
            benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase_sad), inner_dir="sad_text")
        except KeyError:
            Log().logger.error("The key 'sad' was not found in the JSON file.")
    
    
    if selected_task == "5":
        # Image #
        benchmark_manager.register_job(img_f=ReplaceBackgroundImageFilter(image_amanda), inner_dir="RB_amanda")
        benchmark_manager.register_job(img_f=ReplaceBackgroundImageFilter(image_angry), inner_dir="RB_angry")
        benchmark_manager.register_job(img_f=ReplaceBackgroundImageFilter(image_relax), inner_dir="RB_relax")

    if selected_task == "6":
        benchmark_manager.register_job(img_f=PushTopImageFilter(image_amanda), inner_dir="PT_amanda")
        benchmark_manager.register_job(img_f=PushTopImageFilter(image_angry), inner_dir="PT_angry")
        benchmark_manager.register_job(img_f=PushTopImageFilter(image_relax), inner_dir="PT_relax")
    
    # -- Personal information filters -- #
    if selected_task == "7":
        # Text #
        benchmark_manager.register_job(text_f=SurroundByCorrectAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6
        benchmark_manager.register_job(text_f=SurroundByWrongAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6)
        benchmark_manager.register_job(text_f=SurroundByPartialCorrectAnsTextFilter()) #  p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 6
    
    if selected_task == "8":
        # Image #
        benchmark_manager.register_job(img_f=SurroundByCorrectAnsImageFilter(font_path=slurm_font_path)) # num_repeats: int = 5, alpha: float = 0.2,  font_size: int = 40, font_type: str = "arial.ttf", font_color = "black"
        benchmark_manager.register_job(img_f=SurroundByWrongAnsImageFilter(font_path=slurm_font_path)) # same
        benchmark_manager.register_job(img_f=SurroundByPartialCorrectAnsImageFilter(font_path=slurm_font_path)) # p = 0.2, the rest as SurroundByCorrectAnsImageFilter
    
    # --------------------------------------- #

    benchmark_manager.start_workers()
    
    ##########################################################################
    

def shuffle_txt_in_img_eval():
    
    if selected_eval != "B":
        return
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    text_filter = ShuffleWordTextFilter()
    text2image=FilteredFixedSizeText2Image(text_filter, font_path=slurm_font_path)
    
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
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    benchmark_manager.register_job(text_f=ShuffleWordTextFilter(), img_f=IdentityImageFilter())
    
    benchmark_manager.start_workers()
    
    
def shuffle_p_increase_eval():
    
    if selected_eval != "C":
        return
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
    ## Without slurm:
    # text2image = FixedSizeText2Image()
    
    datasetWrapper = GSM8kWrapper(text2image, cache_filename="gsm8k_dataset")
    
    metadata = {
        "test name": "shuffle_p_increase eval",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
        "Save Predictions": True,
    }
    
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    if selected_task == "1":
        # Image #
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.05), inner_dir="SW_p_0_05")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.1), inner_dir="SW_p_0_1")
    
    if selected_task == "2":
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.1), inner_dir="SW_p_0_15")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.4), inner_dir="SW_p_0.25")
        
    if selected_task == "3":
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.4), inner_dir="SW_p_0.3")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.4), inner_dir="SW_p_0.35")
        
    benchmark_manager.start_workers()
        