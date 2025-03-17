import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class AbstractImageFilter:
    def __init__(self, filter_name: str):
        self.filter_name = filter_name
    
    def apply_filter(self, input: Image, answer: str=None):
        return None


class AbstractTextFilter:
    
    def __init__(self, filter_name: str):
        self.filter_name = filter_name
    
    def apply_filter(self, input: str, answer: str=None):
        return None


# ---------- Identity Filters ---------- #


class IdentityTextFilter(AbstractTextFilter):
    
    def __init__(self, filter_name:str="IdentityTextFilter"):
        super().__init__(filter_name)
    
    def apply_filter(self, input: str , answer: str=None):
        return input


class IdentityImageFilter(AbstractImageFilter):
    
    def __init__(self, filter_name:str="IdentityImageFilter"):
        super().__init__(filter_name)
        
    def apply_filter(self, input: Image , answer: str=None):
        return input


# ---------- Noise Filters ---------- #
class ContrastStretchingImageFilter(AbstractImageFilter):
    
    def __init__(self, filter_name:str="ContrastStretchingImageFilter"):
        super().__init__(filter_name)
        
    def apply_filter(self, input: Image , answer: str=None):
        image_array = np.array(input)
        min_pixel = np.min(image_array)
        max_pixel = np.max(image_array)
        stretched_array = ((image_array - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
        return Image.fromarray(stretched_array)
    

class HistogramEqualizationImageFilter(AbstractImageFilter):
    
    def __init__(self, filter_name:str="HistogramEqualizationImageFilter"):
        super().__init__(filter_name)
        
    def apply_filter(self, input: Image , answer: str=None):
        gray_image = input.convert("L")
        image_array = np.array(gray_image)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        equalized_array = cv2.equalizeHist(image_array)
        return Image.fromarray(equalized_array)
    

class GaussianImageFilter(AbstractImageFilter):
    
    def __init__(self, kernel_size: int=5, sigma: float=1.0, filter_name:str="GaussianImageFilter"):
        super().__init__(filter_name)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def apply_filter(self, input: Image , answer: str=None):
        image_array = np.array(input)
        blurred_array = cv2.GaussianBlur(image_array, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(blurred_array) 


class ShuffleWordTextFilter(AbstractTextFilter):
    
    def __init__(self, p: float=0.2, filter_name:str="ShuffleWordTextFilter"):
        super().__init__(filter_name)
        self.p = p
        
    def apply_filter(self, input: str , answer: str=None):
        def scramble_word(word):
            if len(word) > 1 and random.random() < self.p:  
                word_list = list(word)  
                random.shuffle(word_list)  
                return ''.join(word_list) 
            return word

        words = input.split()
        scrambled_words = [scramble_word(word) for word in words]
        return ' '.join(scrambled_words)


class SwapWordsTextFilter(AbstractTextFilter):
    
    def __init__(self, p: float=0.2, filter_name:str="SwapWordsTextFilter"):
        super().__init__(filter_name)
        self.p = p
        
    def apply_filter(self, input: str , answer: str=None):
        words = input.split()
        n = len(words)
        for i in range(n):
            if random.random() < self.p:
                j = random.randint(0, n - 1)
                words[i], words[j] = words[j], words[i]

        return ' '.join(words)


# ---------- General Information Filters ---------- #
class ReplaceBackgroundTextFilter(AbstractImageFilter):

    def __init__(self, background_image: Image , filter_name:str="ReplaceBackgroundTextFilter"):
            super().__init__(filter_name)
            self.background_image: Image = background_image

        
    def apply_filter(self, input: Image , answer: str=None):
            input = np.array(input.convert("L"))  
            bg_array = np.array(self.background_image.convert("RGB"))
            bg_array = cv2.resize(bg_array, (input.size[1], input.size[0]))
            _, mask = cv2.threshold(input, 200, 255, cv2.THRESH_BINARY_INV)  
            mask_3ch = cv2.merge([mask, mask, mask])
            result = np.where(mask_3ch == 255, (0, 0, 0), bg_array)  
            return Image.fromarray(result)


class PushFrontTextFilter(AbstractTextFilter):
    
    def __init__(self, filter_name:str="PushFrontTextFilter"):
        super().__init__(filter_name)
        
    def __init__(self, phrase: str):
        self.phrase = phrase
        
    def apply_filter(self, input: str , answer: str=None):
        return self.phrase + '\n\n' + input


class PushTopImageFilter(AbstractImageFilter):
    def __init__(self, additional_image: Image, filter_name: str="PushTopImageFilter"):
        super().__init__(filter_name)
        self.additional_image = additional_image

    def apply_filter(self, input: Image , answer: str=None):
        width = max(input.width, self.additional_image.width)
        top_resized = self.additional_image.resize((width, int(self.additional_image.height * (width / self.additional_image.width))), Image.LANCZOS)
        input_resized = input.resize((width, int(input.height * (width / input.width))), Image.LANCZOS)
        new_height = top_resized.height + input_resized.height
        combined_image = Image.new("RGB", (width, new_height))
        combined_image.paste(top_resized, (0, 0))
        combined_image.paste(input_resized, (0, top_resized.height))

        return combined_image


# ---------- Personal Information Filters ---------- #
class SurroundByCorrectAnsImageFilter(AbstractImageFilter):
     def __init__(self, num_repeats: int = 10, filter_name: str="SurroundByCorrectAnsImageFilter"):
         super().__init__(filter_name)
         self.num_repeats = num_repeats

     def apply_filter(self, input: Image, answer: str):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.load_default()
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 20)
            draw.text((x, y), answer, font=font, fill= "black")
        return Image.blend(input.convert("RGB"), background, alpha=0.2)
     

class SurroundByWrogAnsImageFilter(AbstractImageFilter):
    RANDOM_VALUES = [
            "TRUE", 3.14, "FALSE", "Bob", 2.71, 99.99, 1492, "TRUE", "FALSE", "TRUE",
            "FALSE", "Charlie", "TRUE", 21, 4096, "FALSE", "Quantum", 7, "FALSE",
            35, 99, "TRUE", "Emma", 1.618, "FALSE", "TRUE", "FALSE", 6.626, "TRUE",
            1969, "Omega", "Alice", "TRUE", "FALSE", 2012, "FALSE", "TRUE", "Delta",
            "Matrix", 512, "TRUE", "FALSE", 99.99, "FALSE", "TRUE", 1999, 256, 1.414,
            "TRUE", "FALSE", "TRUE", "FALSE", "David", "TRUE", "FALSE", "Echo", 13,
            1776, "FALSE", "TRUE", "FALSE", 1865, "FALSE", "TRUE", "Ava", 35, 1789,
            "FALSE", "TRUE", "FALSE", 2012, "FALSE", "TRUE", "Sigma", "FALSE", 1024,
            "FALSE", "TRUE", "FALSE", 1955, "FALSE", "TRUE", "TRUE", 42.42, "FALSE",
            1984, "FALSE", "TRUE", "FALSE", "Noah", "FALSE", "TRUE", "FALSE", 4.669,
            "TRUE", "FALSE", 256, "TRUE", "FALSE", "TRUE", "FALSE", "Sophia", 1789,
            "FALSE", "TRUE", "FALSE", 49, "Liam", 56, 2048, "TRUE", "FALSE", 1984,
            "FALSE", "TRUE", "FALSE", "Blue", 2012, "FALSE", "TRUE", 78, "FALSE",
            "TRUE", "FALSE", "TRUE", "FALSE", "FALSE", 9.81, "TRUE", "FALSE", "Alpha"
        ]
    def __init__(self, num_repeats: int = 10, filter_name: str="SurroundByCorrectAnsImageFilter"):
         super().__init__(filter_name)
         
    def apply_filter(self, input: Image , answer: str=None):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.load_default()
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 20)
            draw.text((x, y), random.shuffle(self.RANDOM_VALUES), font=font, fill= "black")
        return Image.blend(input.convert("RGB"), background, alpha=0.2)


class SurroundByPartialCorrectAnsImageFilter(AbstractImageFilter):
    def __init__(self, p: float = 0.2, num_repeats: int = 10, filter_name:str = "SurroundByPartialCorrectAnsImageFilter"):
        super().__init__(filter_name)
        self.RANDOM_VALUES = SurroundByWrogAnsImageFilter.RANDOM_VALUES
        self.p = p
        self.num_repeats = num_repeats

    def apply_filter(self, input: Image, answer: str):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.load_default()
        current_ans = ""
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 20)
            if random.random() < self.p:  
                current_ans = str(random.choice(self.RANDOM_VALUES))
            else:
                current_ans = answer
            draw.text((x, y), current_ans, font=font, fill= "black")
        return Image.blend(input.convert("RGB"), background, alpha=0.2)
    
        
class SurroundByCorrectAnsTextFilter(AbstractTextFilter):
    def __init__(self, padding_symbol: str = "*", num_repeats: int = 5, filter_name:str = "SurroundByCorrectAnsTextFilter"):
        super().__init__(filter_name)
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str, answer: str ):
        for _ in range(self.num_repeats):
            input = f"{self.padding_symbol}{answer}{self.padding_symbol} {input} {self.padding_symbol}{answer}{self.padding_symbol}"
        return input
    

class SurroundByWrongAnsTextFilter(AbstractTextFilter):
    def __init__(self, padding_symbol: str = "*", num_repeats: int = 5, filter_name:str = "SurroundByWrongAnsTextFilter"):
        super().__init__(filter_name)
        self.RANDOM_VALUES = SurroundByWrogAnsImageFilter.RANDOM_VALUES
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str , answer: str=None):
        for _ in range(self.num_repeats):
            wrong_ans = str(random.choice(self.RANDOM_VALUES))
            input = f"{self.padding_symbol}{wrong_ans}{self.padding_symbol} {input} {self.padding_symbol}{wrong_ans}{self.padding_symbol}"
        return input
    

class SurroundByPartialCorrectAnsTextFilter(AbstractTextFilter):
    def __init__(self, p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 5, filter_name:str = "SurroundByPartialCorrectAnsTextFilter"):
        super().__init__(filter_name)
        self.RANDOM_VALUES = SurroundByWrogAnsImageFilter.RANDOM_VALUES
        self.p = p
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str, answer: str):
        for _ in range(self.num_repeats):
            answer = str(random.choice(self.RANDOM_VALUES)) if random.random() < self.p else answer
            input = f"{self.padding_symbol}{answer}{self.padding_symbol} {input} {self.padding_symbol}{answer}{self.padding_symbol}"
        return input
