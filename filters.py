import cv2
from PIL import Image
import numpy as np
import random

class AbstractImageFilter:
    def apply_filter(self, input: Image):
        return None


class AbstractTextFilter:
    def apply_filter(self, input: str):
        return None


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


class TextBackgroundReplacementFilter(AbstractImageFilter):
    def apply_filter(self, text_image: Image, background_image: Image):
        text_array = np.array(text_image.convert("L"))  
        bg_array = np.array(background_image.convert("RGB")) 
        bg_array = cv2.resize(bg_array, (text_array.shape[1], text_array.shape[0]))
        _, mask = cv2.threshold(text_array, 200, 255, cv2.THRESH_BINARY_INV)  
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, (0, 0, 0), bg_array)  
        return Image.fromarray(result)


class ScrambleLetterInWordsTextFilter(AbstractTextFilter):
    def apply_filter(self, input: str, scramble_probability: float=0.2):
        def scramble_word(word):
            if len(word) <= 1 or random.random() > scramble_probability:  
                return word
            return random.shuffle(word)

        words = input.split()
        scrambled_words = [scramble_word(word) for word in words]
        return ' '.join(scrambled_words)


class ScrambleWordsInSentenceTextFilter(AbstractTextFilter):
    def apply_filter(self, input: str, scramble_probability: float=0.2):
        words = input.split()
        num_words = len(words)
        if num_words < 2:  
            return input
        for _ in range(num_words):  
            if random.random() < scramble_probability:
                idx1, idx2 = random.sample(range(num_words), 2)  
                words[idx1], words[idx2] = words[idx2], words[idx1]  

        return ' '.join(words)


class PushFrontPhraseTextFilter(AbstractTextFilter):
    def __init__(self, phrase: str):
        self.phrase = phrase
        
    def apply_filter(self, input: str):
        return self.phrase + '\n\n' + input
