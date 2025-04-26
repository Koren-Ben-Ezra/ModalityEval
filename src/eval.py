import json
import os

from src.text2image import FixedSizeText2Image, FilteredFixedSizeText2Image
from src.benchmarkManager import BenchmarkManager
from src.filters import *
from src.datasetWrapper import GSM8KWrapper
from src.multimodalWrappers import LlamaWrapper
from src.log import Log

PARAMETERS_PATH = "parameters.json"
slurm_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

selected_eval = os.getenv("SELECTED_EVAL", "0")
selected_task = os.getenv("SELECTED_TASK", "0")

image_path_amanda = os.path.join("images", "amanda.jpg")
image_path_angry = os.path.join("images", "angry.jpg")
image_path_relax = os.path.join("images", "relax.jpg")

image_amanda = Image.open(image_path_amanda)
image_angry = Image.open(image_path_angry)
image_relax = Image.open(image_path_relax)


# Summary of the tests
# A: Basic evaluation
#    # JOB 1: Identity Image Filter + Identity Text Filter + CoT
#    # JOB 2: Identity Image Filter + Identity Text Filter + without CoT

# B: flip2LettersTextFilter text eval
#     # JOB 1: flip2LettersTextFilter p=0.05 p=0.1 p=0.15
#     # JOB 2: flip2LettersTextFilter p=0.2 p=0.25
#     # JOB 3: flip2LettersTextFilter p=0.3 p=0.35

# C: flip2LettersTextFilter Image eval
#     # JOB 1: flip2LettersTextFilter p=0.05
#     # JOB 2: flip2LettersTextFilter p=0.1
#     # JOB 3: flip2LettersTextFilter p=0.15
#     # JOB 4: flip2LettersTextFilter p=0.25
#     # JOB 5: flip2LettersTextFilter p=0.3
#     # JOB 6: flip2LettersTextFilter p=0.35

# D: shuffle_p_increase image eval
#     # JOB 1: shuffle_p_increase image p=0.05
#     # JOB 2: shuffle_p_increase image p=0.1
#     # JOB 3: shuffle_p_increase image p=0.15
#     # JOB 4: shuffle_p_increase image p=0.25
#     # JOB 5: shuffle_p_increase image p=0.3
#     # JOB 6: shuffle_p_increase image p=0.35

# E: shuffle_p_increase text eval
#     # JOB 1: shuffle_p_increase text p=0.05 p=0.1 p=0.15
#     # JOB 2: shuffle_p_increase text p=0.2 p=0.25
#     # JOB 3: shuffle_p_increase text p=0.3 p=0.35

# F: Simple Noise filter for image
#     # JOB 1: Noise filters- Histogram filter for image, Gaussian filter for image
#     # JOB 2: Personalized information filters- Image
#     # JOB 3: Personalized information filters- Text

# G: personalized_information_text_eval
#     # JOB 1: PushFrontImageFilter(text1), PushFrontImageFilter(text2), PushFrontImageFilter(text3)
# H: personalized_information_image_eval
#     # JOB 1: PushFrontImageFilter(text1), PushFrontImageFilter(text2), PushFrontImageFilter(text3)


def with_and_without_cot_instruction_eval():
    if selected_eval != "A":
        return
    
    logger = Log().logger
    with open(PARAMETERS_PATH, 'r') as file:
        parameters = json.load(file)
        
    txt_instruction = ""
    img_instruction = ""
    if selected_task == "1":
        img_instruction = parameters['system']['CoT image instruction']
        txt_instruction = parameters['system']['CoT text instruction']
        
    elif selected_task == "2":
        img_instruction = parameters['system']['image instruction']
        txt_instruction = parameters['system']['text instruction']
    else:
        raise ValueError("execute with: 'sbatch run_slurm.sh A <1-4>'")
    
    logger.info("Starting evaluation...")
    multimodal_wrapper = LlamaWrapper(txt_instruction=txt_instruction, img_instruction=img_instruction)
    text2image=FixedSizeText2Image(font_path=slurm_font_path) 
    datasetWrapper = GSM8KWrapper(text2image)
    
    metadata = {
        "test name": "A test",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    
    dir_name = "with_cot" if selected_task == "2" else "without_cot"
    benchmark_manager.register_job(img_f=IdentityImageFilter(), inner_dir=dir_name)
    benchmark_manager.start_workers()

def flip2LettersTextFilter_TF_eval():
    
    if selected_eval != "B":
        return
    if not ("1" <= selected_task <= "7"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-7>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
    datasetWrapper = GSM8KWrapper(text2image, cache_filename="gsm8k_dataset")
    
    metadata = {
        "test name": "B test: flip2LettersTextFilter text eval",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    if selected_task == "1":
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.05), inner_dir="flip_TF_p_0_05")
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.1), inner_dir="flip_TF_p_0_1")
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.15), inner_dir="flip_TF_p_0_15")
    elif selected_task == "2":
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.2), inner_dir="flip_TF_p_0_2")
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.25), inner_dir="flip_TF_p_0_25")
    elif selected_task == "3":
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.3), inner_dir="flip_TF_p_0_3")
        benchmark_manager.register_job(text_f=flip2LettersTextFilter(p=0.35), inner_dir="flip_TF_p_0_35")  
        
    benchmark_manager.start_workers()

def flip2LettersTextFilter_IF_eval():
    
    if selected_eval != "C":
        return
    if not ("1" <= selected_task <= "7"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-7>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    p_lst: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # selected_task = "1" -> 0.05
    # selected_task = "2" -> 0.1
    # selected_task = "3" -> 0.15
    # selected_task = "4" -> 0.25
    # selected_task = "5" -> 0.3
    # selected_task = "6" -> 0.35
    
    def run_eval(p: float):
        p_str = str(p).replace(".", "_")
        
        filter_p = flip2LettersTextFilter(p=p)
        text2image=FilteredFixedSizeText2Image(filter=filter_p, font_path=slurm_font_path)
        cache_name = f"gsm8k_dataset_p_{p_str}"
        
        dataset_wrapper = GSM8KWrapper(text2image, cache_filename=cache_name)

        metadata = {
            "test name": "C test: flip2LettersTextFilter eval image eval",
            "Model": multimodal_wrapper.model_name,
            "Dataset": dataset_wrapper.dataset_id,
        }
        
        benchmark_manager = BenchmarkManager(dataset_wrapper, multimodal_wrapper, metadata)
        Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {dataset_wrapper.dataset_id} | {selected_eval} | {selected_task}")
        benchmark_manager.register_job(img_f=IdentityImageFilter(), inner_dir=f"flip_IF_{p_str}")
        benchmark_manager.start_workers()

        Log().logger.info(f"Finished evaluation flip2LettersTextFilter for p={p}")
        if os.path.exists(cache_name):
            Log().logger.info(f"Removing cache file: {cache_name}")
            os.remove(cache_name)


    for task in range(len(p_lst)):
        if selected_task == str(task + 1):
            selected_p = p_lst[task]
            run_eval(selected_p)
            break

def shuffle_p_increase_image_eval():
    
    if selected_eval != "D":
        return
    if not ("1" <= selected_task <= "7"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-7>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    p_lst: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # selected_task = "1" -> 0.05
    # selected_task = "2" -> 0.1
    # selected_task = "3" -> 0.15
    # selected_task = "4" -> 0.25
    # selected_task = "5" -> 0.3
    # selected_task = "6" -> 0.35
    
    def run_eval(p: float):
        p_str = str(p).replace(".", "_")
        
        filter_p = ShuffleWordTextFilter(p=p)
        text2image=FilteredFixedSizeText2Image(filter=filter_p, font_path=slurm_font_path)
        cache_name = f"gsm8k_dataset_p_{p_str}"
        
        dataset_wrapper = GSM8KWrapper(text2image, cache_filename=cache_name)

        metadata = {
            "test name": "shuffle_p_increase eval",
            "Model": multimodal_wrapper.model_name,
            "Dataset": dataset_wrapper.dataset_id,
        }
        
        benchmark_manager = BenchmarkManager(dataset_wrapper, multimodal_wrapper, metadata)
        Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {dataset_wrapper.dataset_id} | {selected_eval} | {selected_task}")
        benchmark_manager.register_job(img_f=IdentityImageFilter(), inner_dir=f"shuffle_IF_{p_str}")
        benchmark_manager.start_workers()

        Log().logger.info(f"Finished evaluation for p={p}")
        if os.path.exists(cache_name):
            Log().logger.info(f"Removing cache file: {cache_name}")
            os.remove(cache_name)


    for task in range(len(p_lst)):
        if selected_task == str(task + 1):
            selected_p = p_lst[task]
            run_eval(selected_p)
            break

def shuffle_p_increase_text_eval():
    if selected_eval != "E":
        return
    if selected_task == "0":
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <task>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
    datasetWrapper = GSM8KWrapper(text2image, cache_filename="gsm8k_dataset")
    
    metadata = {
        "test name": "E: shuffle_p_increase eval",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    if selected_task == "1":
        # Image #
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.05), inner_dir="SW_p_0_05")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.1), inner_dir="SW_p_0_1")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.15), inner_dir="SW_p_0_15")
    
    if selected_task == "2":
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.2), inner_dir="SW_p_0_2")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.25), inner_dir="SW_p_0.25")
        
    if selected_task == "3":
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.3), inner_dir="SW_p_0.3")
        benchmark_manager.register_job(text_f=ShuffleWordTextFilter(p=0.35), inner_dir="SW_p_0.35")
        
    benchmark_manager.start_workers()

def basic_eval_all():
    if selected_eval != "F":
        return
    if not ("1" <= selected_task <= "3"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-3>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
    datasetWrapper = GSM8KWrapper(text2image)
    
    metadata = {
        "test name": "F: basic Noise test",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    # Read the JSON file
    with open(PARAMETERS_PATH, 'r') as file:
        parameters = json.load(file)
    
    try:
        phrases = parameters['phrases']
    except KeyError:
        Log().logger.error("The key 'phrases' was not found in the JSON file.")
        return

    # Initialize the BenchmarkManager
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    # ------ execute text and image filter tests ------ #
    # -- Noise filters -- #
    if selected_task == "1":
        benchmark_manager.register_job(img_f=HistogramEqualizationImageFilter())
        benchmark_manager.register_job(img_f=GaussianImageFilter()) # kernel_size = 5 sigms = 1.0
    # -- Personalized information filters -- #
    if selected_task == "2":
        benchmark_manager.register_job(text_f=SurroundByCorrectAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6
        benchmark_manager.register_job(text_f=SurroundByWrongAnsTextFilter()) # padding_symbol: str = "*", num_repeats: int = 6)
        benchmark_manager.register_job(text_f=SurroundByPartialCorrectAnsTextFilter()) #  p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 6
    if selected_task == "3":
        benchmark_manager.register_job(img_f=SurroundByCorrectAnsImageFilter(font_path=slurm_font_path)) # num_repeats: int = 5, alpha: float = 0.2,  font_size: int = 40, font_type: str = "arial.ttf", font_color = "black"
        benchmark_manager.register_job(img_f=SurroundByWrongAnsImageFilter(font_path=slurm_font_path)) # same
        benchmark_manager.register_job(img_f=SurroundByPartialCorrectAnsImageFilter(font_path=slurm_font_path)) # p = 0.2, the rest as SurroundByCorrectAnsImageFilter

def personalized_information_text_eval():
    if selected_eval != "G":
        return
    if not ("1" <= selected_task <= "3"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-3>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ## With slurm:
    text2image=FixedSizeText2Image(font_path=slurm_font_path)
    
    datasetWrapper = GSM8KWrapper(text2image)
    
    metadata = {
        "test name": "G: basic Noise test",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    # Read the JSON file
    with open(PARAMETERS_PATH, 'r') as file:
        parameters = json.load(file)
    
    try:
        phrases = parameters['phrases']
    except KeyError:
        Log().logger.error("The key 'phrases' was not found in the JSON file.")
        return

    # Initialize the BenchmarkManager
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase=phrases["military"]))
    benchmark_manager.register_job(text_f=PushFrontTextFilter(phrase=phrases["relax"]))
        
    benchmark_manager.start_workers()
    
def personalized_information_image_eval():
    if selected_eval != "H":
        return
    if not ("1" <= selected_task <= "2"):
        raise ValueError("execute with: 'sbatch run_slurm.sh <eval> <1-2>'")
    
    # prepare the benchmark
    Log().logger.info("------------------------------------------------")
    Log().logger.info("Starting evaluation...")
    
    multimodal_wrapper = LlamaWrapper()
    
    ## With slurm:
    with open(PARAMETERS_PATH, 'r') as file:
        parameters = json.load(file)
        phrases = parameters['phrases']
        
    if selected_task == "1":
        phrase = phrases["military"]
    elif selected_task == "2":
        phrase = phrases["relax"]
        
    text2image=FilteredFixedSizeText2Image(filter=PushFrontTextFilter(phrase), font_path=slurm_font_path)
    
    datasetWrapper = GSM8KWrapper(text2image, cache_filename=f"gsm8k_dataset_phrase_{selected_task}")
    
    metadata = {
        "test name": "H",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
    }
    # Read the JSON file
    with open(PARAMETERS_PATH, 'r') as file:
        parameters = json.load(file)
    
    try:
        phrases = parameters['phrases']
    except KeyError:
        Log().logger.error("The key 'phrases' was not found in the JSON file.")
        return

    # Initialize the BenchmarkManager
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    Log().logger.info(f"Running benchmark: {multimodal_wrapper.model_name} | {datasetWrapper.dataset_id} | {selected_eval} | {selected_task}")
    
    benchmark_manager.register_job(img_f=IdentityImageFilter(), inner_dir=f"push_front_IF_{selected_task}")
    benchmark_manager.start_workers()