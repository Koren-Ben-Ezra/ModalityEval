import textwrap
from datasets import load_dataset, Dataset
from src.text2image import AbstractText2Image, FixedSizeText2Image
from src.log import Log
import os
import json
from PIL import Image, ImageDraw, ImageFont


CACHE_NAME = "cache"

class AbstractDatasetWrapper:
    """
    Abstract base class for dataset wrappers.
    Ensures that each dataset wrapper implements a standardized dataset structure.
    """
    def __init__(self, text2image: AbstractText2Image):
        self.dataset: Dataset = None  # Should be implemented in subclasses
        self.dataset_id = None
        self._text2image = text2image


class GSM8kWrapper_amit(AbstractDatasetWrapper):
    """
    GSM8k wrapper that loads the full test set.
    """
    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image()):
        super().__init__(text2image)
        self.dataset_id = "GSM8k"
        self._text2image = text2image

        Log().logger.info(f"Loading full test set from {self.dataset_id}...")
        try:
            raw = load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            Log().logger.error(f"Error loading GSM8k test set: {e}")
            raise

        questions = [ex["question"] for ex in raw]
        longest_question = max(questions, key=len)

        Log().logger.info(
            f"Longest question (in test set): '{longest_question}' (len={len(longest_question)})"
        )

        self._text2image.find_font_size(longest_question)
        Log().logger.info(f"Font size for longest question: {self._text2image.font_size}")

        try:
            self.dataset = raw.map(self._map_sample, load_from_cache_file=False, keep_in_memory=True)
        except Exception as e:
            Log().logger.error(f"Error mapping GSM8k test set: {e}")
            raise

        Log().logger.info(f"Loaded {len(self.dataset)} samples from {self.dataset_id}.")

    def _map_sample(self, sample):
        sample["answer"] = sample["answer"].split("####")[-1].strip()
        try:
            sample["question_image"] = self._text2image.create_image(sample["question"])
        except Exception as e:
            Log().logger.error(f"Error creating image: {e}")
            raise e

        return sample


class GSM8kWrapper_GSM8k_5_samples(AbstractDatasetWrapper):
    """
    A slimmed-down GSM8k wrapper that loads only 5 examples.
    """
    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image()):
        super().__init__(text2image)
        self.dataset_id = "GSM8k_5_samples"
        self._text2image = text2image

        Log().logger.info(f"Loading 5 examples from {self.dataset_id}...")
        try:
            # load only the first 5 test samples
            raw = load_dataset("gsm8k", "main", split="test[:5]")
        except Exception as e:
            Log().logger.error(f"Error loading 5-sample GSM8k: {e}")
            raise

        # find the question with the maximum length among these 5
        questions = [ex["question"] for ex in raw]
        longest_question = max(questions, key=len)

        Log().logger.info(
            f"Longest question (of 5): '{longest_question}' (len={len(longest_question)})"
        )

        self._text2image.find_font_size(longest_question)
        Log().logger.info(f"Font size for longest question: {self._text2image.font_size}")

        # apply mapping (answer cleanup + image creation)
        try:
            self.dataset = raw.map(self._map_sample,load_from_cache_file=False,keep_in_memory=True)
        except Exception as e:
            Log().logger.error(f"Error mapping 5-sample GSM8k: {e}")
            raise

        Log().logger.info(f"Loaded {len(self.dataset)} samples from {self.dataset_id}.")

    def _map_sample(self, sample):
        sample["answer"] = sample["answer"].split("####")[-1].strip()
        try:
            sample["question_image"] = self._text2image.create_image(sample["question"])
        except Exception as e:
            Log().logger.error(f"Error creating image: {e}")
            raise e

        return sample

    


class GSM8KWrapper(AbstractDatasetWrapper):

    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image(), cache_filename: str=""):
        self.dataset_id = "GSM8k"
        self._text2image = text2image
        Log().logger.info(f"Loading {self.dataset_id} dataset...")
        
        if not os.path.exists(CACHE_NAME):
            os.makedirs(CACHE_NAME)
        
        if not cache_filename:
            cache_filename = f"{self.dataset_id}_dataset"
            
        cache_path = os.path.join(CACHE_NAME, cache_filename)
        if os.path.exists(cache_path):
            Log().logger.info(f"Found cached dataset at {cache_path}. Loading from cache...")
            self.dataset = Dataset.load_from_disk(str(cache_path))
            return
        
        else:
            try:
                self.dataset = load_dataset("gsm8k", "main")["test"]
            except Exception as e:
                Log().logger.error(f"Error loading dataset: {e}")
                raise e

            try:
                longest_question_sample = max(self.dataset, key=lambda x: len(x['question']))
                longest_question = longest_question_sample['question']

                self._text2image.find_font_size(longest_question)
                Log().logger.info(f"Font size for longest question: {self._text2image.font_size}")

                
                self.dataset = self.dataset.map(self._map_sample)

            except Exception as e:
                Log().logger.error(f"Error mapping dataset: {e}")
                raise e
            
            try:
                self.dataset.save_to_disk(cache_path)
                Log().logger.info(f"Cached dataset at {cache_path}")
            except Exception as e:
                Log().logger.error(f"Error caching dataset: {e}")
                raise e
        
        Log().logger.info(f"Loaded {self.dataset_id} dataset with {len(self.dataset)} samples.")
        
    def _map_sample(self, sample):
        sample["answer"] = sample["answer"].split("####")[-1].strip()
            
        try:
            sample["question_image"] = self._text2image.create_image(sample["question"])
        except Exception as e:
            Log().logger.error(f"Error creating image: {e}")
            raise e
        return sample

class GSM8kWrapper_old(AbstractDatasetWrapper):

    def __init__(self, text2image: AbstractText2Image = FixedSizeText2Image(), cache_filename: str=""):
        self.dataset_id = "GSM8k"

        # Load the GSM8K dataset
        with open('gsm8k_train.jsonl', 'r') as f:
            data = [json.loads(line) for line in f]


        

        # Find the question with the maximum length
        self.longest_question = max(data, key=lambda x: len(x['question'])) 
        Log().logger.info(f"Longest question found in  dataset: {self.longest_question['question']} with length {len(self.longest_question['question'])}")

        self._text2image = text2image
        Log().logger.info(f"Loading {self.dataset_id} dataset...")
        
        if not os.path.exists(CACHE_NAME):
            os.makedirs(CACHE_NAME)
        
        if not cache_filename:
            cache_filename = f"{self.dataset_id}_dataset"
            
        cache_path = os.path.join(CACHE_NAME, cache_filename)
        if os.path.exists(cache_path):
            Log().logger.info(f"Found cached dataset at {cache_path}. Loading from cache...")
            self.dataset = Dataset.load_from_disk(str(cache_path))
            return
        
        else:
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
            
            try:
                self.dataset.save_to_disk(cache_path)
                Log().logger.info(f"Cached dataset at {cache_path}")
            except Exception as e:
                Log().logger.error(f"Error caching dataset: {e}")
                raise e
        
        Log().logger.info(f"Loaded {self.dataset_id} dataset with {len(self.dataset)} samples.")
        
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
