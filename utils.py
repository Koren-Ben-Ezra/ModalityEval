import cv2
from PIL import Image, ImageFilter
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq

class MultimodalWrapper:
    def generate(self, text: str, img: Image):
        return None
    
class Llama32Wrapper(MultimodalWrapper):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
        self.model = AutoModelForVision2Seq.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

    def generate(self, text: str, img: Image):
        return self.model.generate(text, img)
    
    
class ModelManager:
    def __init__(self, multimodal: MultimodalWrapper):
        self._multimodal = multimodal

    def separate_forward(self, text_q: str, img_q: Image, true_answer: str):
        answer_text_q = self._forward_text(text_q)
        answer_img_q = self._forward_image(img_q)
        return answer_text_q, answer_img_q

    def _forward_image(self, img_q: Image):
        pass

    def _forward_text(self, text_q: str):
        pass


class AbstractImageFilter:
    def apply_filter(self, input: Image):
        return None


class AbstractTextFilter:
    def apply_filter(self, input: str):
        return None


class IdentityTextFilter(AbstractTextFilter):
    def apply_filter(self, input: str):
        return input


class IdentityImageFilter(AbstractImageFilter):
    def apply_filter(self, input: Image):
        return input

class GaussianImageFilter(AbstractImageFilter):
    def apply_filter(self, input: Image, kernel_size: int=5, sigma: float=1.0):
        image_array = np.array(input)
        blurred_array = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), sigma)
        return Image.fromarray(blurred_array) 

class ContrastStretchingImageFilter(AbstractImageFilter):
    def apply_filter(self, input: Image):
        image_array = np.array(input)
        min_pixel = np.min(image_array)
        max_pixel = np.max(image_array)
        stretched_array = ((image_array - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
        return Image.fromarray(stretched_array)

class HistogramEqualizationImageFilter(AbstractImageFilter):
    def apply_filter(self, input: Image):
        image_array = np.array(input)
        equalized_array = cv2.equalizeHist(image_array)
        return Image.fromarray(equalized_array)

class Category:
    class Statistics:
        def __init__(self):
            self.success = 0
            self.failures = 0


    def __init__(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter):
        self.text_f = text_f
        self.img_f = img_f
        self.text_stats = Category.Statistics()
        self.img_stats = Category.Statistics()
