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
        self.lognest_question = None
        self._text2image = text2image

        
    def find_font_size(self, text: str, longest_text: int = 500):
        
        try:
            text = AbstractText2Image._preprocess_text(text)
        except Exception as e:
            Log().logger.error(f"Error in preprocessing text: {e}")
            raise e
        
        W, H = self.width, self.height
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        longest_text = text
        lo, hi = 5, 300
        best_font_size = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid) if font_path else ImageFont.load_default()
            draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

            avg_char_width = sum(draw.textsize(c, font=font)[0] for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") / 52
            chars_per_line = max(1, int(W / avg_char_width))
            wrapped = textwrap.wrap(longest_text, width=chars_per_line)

            ascent, descent = font.getmetrics()
            line_height = int((ascent + descent) * 1.15)
            total_height = line_height * len(wrapped)

            if total_height <= H:
                best_font_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best_font_size



class GSM8kWrapper(AbstractDatasetWrapper):
    def __init__(self, text2image=None):
        self.dataset_id = "GSM8k"
        self._text2image = text2image

        self.data = [
            {"question": "Tom has 3 pencils and buys 7 more. How many does he have now?"},
            {"question": "A train travels at 60 km/h. How far does it travel in 4 hours?"},
            {"question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
            {"question": "If a square has side length 5 cm, what is its area?"},
            {"question": "Sam reads 10 pages every day. How many pages does he read in a week?"}
        ]

        self.longest_question = max(self.data, key=lambda x: len(x['question']))
        self.font_size = self.find_font_size(self.longest_question)
        print(f"Longest question found: {self.longest_question['question']} (length={len(self.longest_question['question'])})")

        # Dummy dataset creation
        self.dataset = Dataset.from_list([self._map_sample(sample) for sample in self.data])

    def _map_sample(self, sample):
        sample["answer"] = "dummy_answer"
        sample["question_image"] = self._text2image.create_image(sample["question"], self.font_size) if self._text2image else None
        return sample

    
class GSM8kWrapper_2(AbstractDatasetWrapper):

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
