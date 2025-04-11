import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

RANDOM_VALUES: list[str] = [str(random.randint(0, 10000)) for _ in range(100)]

class AbstractImageFilter:
    filter_name = None
    
    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     FilterLoader.register_image_filter(cls)
        
    def apply_filter(self, input: Image, answer: str=None):
        return None

class AbstractTextFilter:
    filter_name = None
    
    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     FilterLoader.register_text_filter(cls)
    
    def apply_filter(self, input: str, answer: str=None):
        return None

# class FilterLoader:

#     image_filters = {}
#     text_filters = {}
        
#     @classmethod
#     def register_image_filter(cls, image_filter_cls: AbstractImageFilter):
#         cls.image_filters[image_filter_cls.filter_name] = image_filter_cls
                
#     @classmethod
#     def register_text_filter(cls, text_filter_cls: AbstractTextFilter):
#         cls.text_filters[text_filter_cls.filter_name] = text_filter_cls


# ---------- Identity Filters ---------- #


class IdentityTextFilter(AbstractTextFilter):
    filter_name:str="Identity_TF"

    def apply_filter(self, input: str , answer: str=None):
        return input


class IdentityImageFilter(AbstractImageFilter):
    filter_name:str="Identity_IF"
    
    def apply_filter(self, input: Image , answer: str=None):
        return input


# ---------- Noise Filters ---------- #
class ContrastStretchingImageFilter(AbstractImageFilter):
    filter_name:str="ContrastStretching_IF"
        
    def apply_filter(self, input: Image , answer: str=None):
        image_array = np.array(input)
        min_pixel = np.min(image_array)
        max_pixel = np.max(image_array)
        stretched_array = ((image_array - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
        return Image.fromarray(stretched_array)
    

class HistogramEqualizationImageFilter(AbstractImageFilter):
    filter_name:str="HistogramEqualization_IF"
        
    def apply_filter(self, input: Image , answer: str=None):
        gray_image = input.convert("L")
        image_array = np.array(gray_image)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        equalized_array = cv2.equalizeHist(image_array)
        return Image.fromarray(equalized_array)
    

class GaussianImageFilter(AbstractImageFilter):
    
    filter_name:str="Gaussian_IF"
    
    def __init__(self, kernel_size: int=5, sigma: float=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def apply_filter(self, input: Image , answer: str=None):
        image_array = np.array(input)
        blurred_array = cv2.GaussianBlur(image_array, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(blurred_array) 


class ShuffleWordTextFilter(AbstractTextFilter):
    filter_name:str="ShuffleWord_TF"
    def __init__(self, p: float=0.2):
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
    filter_name:str="SwapWords_TF"
    def __init__(self, p: float=0.2):
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
class ReplaceBackgroundImageFilter(AbstractImageFilter):
    filter_name:str="ReplaceBackground_IF"
    def __init__(self, background_image: Image , alpha: float=0.5):
        self.bg_image: Image = background_image
        self.alpha = alpha
        
    def apply_filter(self, input: Image , answer: str=None):
        bg_image_resized = self.bg_image.resize(input.size, Image.LANCZOS) 
        return Image.blend(input.convert("RGB"), bg_image_resized.convert("RGB"), alpha=self.alpha)


class PushFrontTextFilter(AbstractTextFilter):
    filter_name:str="PushFront_TF"
    def __init__(self, phrase: str):
        self.phrase = phrase
        
    def apply_filter(self, input: str , answer: str=None):
        return self.phrase + '\n\n' + input


class PushTopImageFilter(AbstractImageFilter):
    filter_name: str="PushTop_IF"
    def __init__(self, additional_image: Image):
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
    filter_name: str="SurroundByCorrectAns_IF"
    def __init__(
        self, 
        num_repeats: int = 5, 
        alpha: float = 0.2, 
        font_size: int = 40, 
        font_type: str = "arial.ttf", 
        font_color = "black"):
        
        self.num_repeats = num_repeats
        self.font_size = font_size
        self.font_type = font_type
        self.font_color = font_color
        self.alpha = alpha

    def apply_filter(self, input: Image, answer: str):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.truetype(self.font_type, self.font_size)
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 20)
            draw.text((x, y), answer, font=font, fill= self.font_color)
        return Image.blend(input.convert("RGB"), background, alpha=self.alpha)
    

class SurroundByWrongAnsImageFilter(AbstractImageFilter):
    filter_name: str="SurroundByWrongAns_IF"
    def __init__(
            self, 
            num_repeats: int = 5, 
            alpha: float = 0.2, 
            font_size: int = 40, 
            font_type: str = "arial.ttf", 
            font_color = "black"):
        
        self.num_repeats = num_repeats
        self.font_size = font_size
        self.font_type = font_type
        self.font_color = font_color
        self.alpha = alpha
        
    def apply_filter(self, input: Image , answer: str=None):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.truetype(self.font_type, self.font_size)
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 50)
            draw.text((x, y), random.choice(RANDOM_VALUES), font=font, fill= self.font_color)
        return Image.blend(input.convert("RGB"), background, alpha=self.alpha)


class SurroundByPartialCorrectAnsImageFilter(AbstractImageFilter):
    filter_name: str="SurroundByPartialCorrectAns_IF"
    def __init__(
            self, 
            p: float = 0.2,
            num_repeats: int = 5, 
            alpha: float = 0.2, 
            font_size: int = 40, 
            font_type: str = "arial.ttf", 
            font_color = "black"):
        
        self.p = p
        self.num_repeats = num_repeats
        self.font_size = font_size
        self.font_type = font_type
        self.font_color = font_color
        self.alpha = alpha

    def apply_filter(self, input: Image, answer: str):
        width, height = input.size
        background = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(background)
        font = ImageFont.truetype(self.font_type, self.font_size)
        current_ans = ""
        for _ in range(self.num_repeats):
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 50)
            if random.random() < self.p:  
                current_ans = random.choice(RANDOM_VALUES) 
            else:
                current_ans = answer
            draw.text((x, y), current_ans, font=font, fill= self.font_color)
        return Image.blend(input.convert("RGB"), background, alpha=self.alpha)
    
        
class SurroundByCorrectAnsTextFilter(AbstractTextFilter):
    filter_name:str = "SurroundByCorrectAns_TF"
    def __init__(self, padding_symbol: str = "*", num_repeats: int = 6):
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str, answer: str ):
        prefix = []
        postfix = []
        for _ in range(self.num_repeats//2):
            # prefix.append(random.choice(RANDOM_VALUES))
            # postfix.append(random.choice(RANDOM_VALUES))
            prefix.append(answer)
            postfix.append(answer)
        prefix = self.padding_symbol + " ".join(prefix) + self.padding_symbol
        postfix = self.padding_symbol + " ".join(postfix) + self.padding_symbol
        output = prefix + "\n" + input + "\n" + postfix
        return output
    

class SurroundByWrongAnsTextFilter(AbstractTextFilter):
    filter_name:str = "SurroundByWrongAns_TF"
    def __init__(self, padding_symbol: str = "*", num_repeats: int = 6):
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str , answer: str=None):
        prefix = []
        postfix = []
        for _ in range(self.num_repeats//2):
            prefix.append(random.choice(RANDOM_VALUES))
            postfix.append(random.choice(RANDOM_VALUES))
        prefix = self.padding_symbol + " ".join(prefix) + self.padding_symbol
        postfix = self.padding_symbol + " ".join(postfix) + self.padding_symbol
        output = prefix + "\n" + input + "\n" + postfix
        return output
    

class SurroundByPartialCorrectAnsTextFilter(AbstractTextFilter):
    filter_name:str = "SurroundByPartialCorrectAns_TF"
    def __init__(self, p: float = 0.2, padding_symbol: str = "*", num_repeats: int = 6):
        self.p = p
        self.padding_symbol = padding_symbol
        self.num_repeats = num_repeats

    def apply_filter(self, input: str, answer: str):
        prefix = []
        postfix = []
        for _ in range(self.num_repeats//2):
            if random.random() < self.p:  
                prefix.append(random.choice(RANDOM_VALUES))
            else:
                prefix.append(answer)
                
            if random.random() < self.p:  
                postfix.append(random.choice(RANDOM_VALUES))
            else:
                postfix.append(answer)
                
        prefix = self.padding_symbol + " ".join(prefix) + self.padding_symbol
        postfix = self.padding_symbol + " ".join(postfix) + self.padding_symbol
        output = prefix + "\n" + input + "\n" + postfix
        return output