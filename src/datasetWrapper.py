from datasets import load_dataset, Dataset
from src.text2image import AbstractText2Image, FixedSizeText2Image

class AbstractDatasetWrapper:
    """
    Abstract base class for dataset wrappers.
    Ensures that each dataset wrapper implements a standardized dataset structure.
    """
    def __init__(self, text2image: AbstractText2Image):
        self.dataset: Dataset = None  # Should be implemented in subclasses
        self.dataset_id = None
        self._text2image = text2image
    
class GSM8kWrapper(AbstractDatasetWrapper):
    '''
    example for element in the gsm8k dataset:

    'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast 
                every morning and bakes muffins for her friends every day with four. 
                She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
                How much in dollars does she make every day at the farmers' market?", 
    'answer':   'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
                She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
                #### 18'
    '''
    
    def __init__(self, text2image: AbstractText2Image=FixedSizeText2Image()):
        self.dataset_id = "GSM8k"
        self._text2image = text2image
        self.dataset = load_dataset("gsm8k", "main")["test"].select(range(10))
        self.dataset = self.dataset.map(self._map_sample)
    
    #separete question, answer calculations and finel numerical answer
    def _map_sample(self, sample):
        # Use the delimiter to extract the ground-truth number
        print("Original sample:", sample,flush=True) 
        parts = sample["answer"].split("####")
        if len(parts) < 2:
            sample["answer"] = sample["answer"].strip()
        else:
            sample["answer"] = parts[-1].strip()
        sample["question_image"] = self._text2image.create_image(sample["question"])
        return sample
    
# TODO: Implement wrappers for other datasets
# class SQuADWrapper(AbstractDatasetWrapper):
#     pass    

# class TriviaQAWrapper(AbstractDatasetWrapper):
#     pass