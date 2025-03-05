from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, Dataset

class AbstractText2Image:
    def create_image(self, text: str):
        return None
    
class Text2ImageDefault(AbstractText2Image):
    def __init__(self, font_name: str="ariel.ttf", background_image: Image=None, font_size=20, padding=10, background_color="white", text_color="black"):
        self.font_name = font_name
        self.font_size = font_size
        self.padding = padding
        self.background_color = background_color
        self.text_color = text_color
        self.background = background_image
        
    def create_image(self, text: str):
        try:
            font = ImageFont.truetype(self.font_name, self.font_size)
        except:
            font = ImageFont.load_default()
    
        # Calculate text size
        lines = text.split("\n")
        line_height = self.font_size + int(0.5 * self.font_size)
        image_height = line_height * len(lines) + self.padding
        # Calculate image width more accurately
        image_width = max(font.getbbox(line)[2] for line in lines) + self.padding
        # Create an image
        img = bg = Image.new("RGB", (image_width, image_height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Draw the text
        y_text = self.padding // 2
        for line in lines:
            draw.text((self.padding//2, y_text), line, self.text_color, font=font)
            y_text += line_height
        
        return img
    
    
    # def _create_fit_background(self, width, height):
    #     if self.background_image:
    #         bg = Image.open(self.background_image).convert("RGB")
    #         bg = bg.resize((width, height), Image.LANCZOS)  # Resize to fit text
    #     else:
    #         bg = Image.new("RGB", (width, height), self.background_color)
        
    #     return bg
    
    
# class Text2ImageCustomBackground(AbstractText2Image):
#     def create_image(self, text: str, background_image: Image = None):
#         return Text2ImageDefault.create_image(text, background_image) 


class AbstractDatasetWrapper:
    """
    Abstract base class for dataset wrappers.
    Ensures that each dataset wrapper implements a standardized dataset structure.
    """
    def __init__(self, text2image: AbstractText2Image):
        self.dataset: Dataset = None  # Should be implemented in subclasses
        self._text2image = text2image
    
class GSM8kWrapper(AbstractDatasetWrapper):
    '''
    example for element in the gsm8k dataset:

    'question': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast 
                every morning and bakes muffins for her friends every day with four. 
                She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
                How much in dollars does she make every day at the farmers' market?", 
    'answer':   'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
                She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
                #### 18'
    '''
    
    def __init__(self, text2image: AbstractText2Image=Text2ImageDefault()):
        self._text2image = text2image
        self.dataset = load_dataset("gsm8k", "main")["test"]
        self.dataset = self.dataset.map(self._map_sample)
        
    def _map_sample(self, sample):
        sample["answer"] = sample["answer"].split("####")[1].strip()
        sample["question_image"] = self._text2image.create_image(sample["question"])
        return sample
    
# TODO: Implement wrappers for other datasets
# class SQuADWrapper(AbstractDatasetWrapper):
#     pass    

# class TriviaQAWrapper(AbstractDatasetWrapper):
#     pass