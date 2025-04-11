from src.text2image import FixedSizeText2Image
from src.benchmarkManager import BenchmarkManager
from src.filters import *
from src.datasetWrapper import GSM8kWrapper
from src.multimodalWrappers import LlamaWrapper

def eval_Llama32vision():
    # prepare the benchmark
    multimodal_wrapper = LlamaWrapper()
    
    ############################# GSM8k dataset ##############################
    
    ## With slurm:
    # text2image=FixedSizeText2Image(font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
    ## Without slurm:
    text2image = FixedSizeText2Image()
    datasetWrapper = GSM8kWrapper(text2image)
    
    metadata = {
        "test name": "demo test with two filters",
        "Model": multimodal_wrapper.model_name,
        "Dataset": datasetWrapper.dataset_id,
        "Save Predictions": True,
        "Use Text Filter": True,
        "Use Image Filter": False 
    }
    
    benchmark_manager = BenchmarkManager(datasetWrapper, multimodal_wrapper, metadata)
    
    # ------ execute text filter tests ------ #
    
    ## shuffle filter test
    shuffle_filter = ShuffleWordTextFilter(p=0.2)
    benchmark_manager.execute_test(text_f=shuffle_filter)
    
    ## random filter test
    swap_words_filter = SwapWordsTextFilter(p=0.2)
    benchmark_manager.execute_test(text_f=swap_words_filter)
    
    # --------------------------------------- #
    
    benchmark_manager.write_summary()
    
    ##########################################################################