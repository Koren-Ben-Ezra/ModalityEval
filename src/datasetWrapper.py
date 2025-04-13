from datasets import load_dataset, Dataset
from text2image import AbstractText2Image, FixedSizeText2Image
from src.log import Log

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

    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image()):
        self.dataset_id = "GSM8k"
        self._text2image = text2image
        Log().logger.info(f"Loading {self.dataset_id} dataset...")
        
        try:
            self.dataset = load_dataset("gsm8k", "main")["test"]
        except Exception as e:
            Log().logger.error(f"Error loading dataset: {e}")
            raise e
        
        try:
            self.dataset = self.dataset.map(self._map_sample)
        except Exception as e:
            Log().logger.error(f"Error mapping dataset: {e}")
            raise e
        
        Log().logger.info(f"Loaded {len(self.dataset)} samples from {self.dataset_id} dataset.")
    
    def _map_sample(self, sample):
        sample["answer"] = sample["answer"].split("####")[-1].strip()
            
        try:
            sample["question_image"] = self._text2image.create_image(sample["question"])
        except Exception as e:
            Log().logger.error(f"Error creating image: {e}")
            raise e
        return sample


class SQuADWrapper(AbstractDatasetWrapper):
    """
    Wrapper for the SQuAD dataset (version 1.1).
    Adjust the split (train/validation) as needed.
    """
    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image()):
        self.dataset_id = "SQuAD"
        self._text2image = text2image
        # Load the dataset. 
        # For example, use the validation split in SQuAD v1.1
        self.dataset = load_dataset("squad")["validation"].select(range(3))
        self.dataset = self.dataset.map(self._map_sample)

    def _map_sample(self, sample):
        # Here, we take the first gold answer text if it exists
        answers = sample.get("answers", {})
        possible_answers = answers.get("text", [])
        sample["answer"] = possible_answers[0] if len(possible_answers) > 0 else ""
        
        # Convert the question text to an image
        sample["question_image"] = self._text2image.create_image(sample["question"])
        return sample


class TriviaQAWrapper(AbstractDatasetWrapper):
    """
    Wrapper for the TriviaQA dataset.
    You may want to confirm whether "rc" or "unfiltered" configuration 
    fits your needs. Example below uses "rc" and the validation split.
    """
    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image()):
        self.dataset_id = "TriviaQA"
        self._text2image = text2image
        # Load the dataset. 
        # For example, use the 'rc' config with the validation split
        self.dataset = load_dataset("trivia_qa", "rc")["validation"].select(range(3))
        self.dataset = self.dataset.map(self._map_sample)

    def _map_sample(self, sample):
        # The 'answer' field is typically a dict with {'value': '...'}
        answer_dict = sample.get("answer", {})
        sample["answer"] = answer_dict.get("value", "")
        
        # Convert the question text to an image
        sample["question_image"] = self._text2image.create_image(sample["question"])
        return sample
