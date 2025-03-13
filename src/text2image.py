from PIL import Image, ImageDraw, ImageFont


MAX_WORDS_PER_LINE__IMG = 10

class AbstractText2Image:
    def create_image(self, text: str):
        return None
    
class Text2ImageDefault(AbstractText2Image):
    def __init__(self, font_name: str="ariel.ttf", background_image: Image=None, font_size=100, padding=5, background_color="white", text_color="black"):
        self.font_name = font_name
        self.font_size = font_size
        self.padding = padding
        self.background_color = background_color
        self.text_color = text_color
        self.background = background_image
        
    def _preprocess_text(self, text: str):
        lines = text.split("\n")
        reshaped_lines = []
        for line in lines:
            words = line.split(" ")
            while len(words) > MAX_WORDS_PER_LINE__IMG:
                reshaped_lines.append(" ".join(words[:MAX_WORDS_PER_LINE__IMG]))
                words = words[MAX_WORDS_PER_LINE__IMG:]
            reshaped_lines.append(" ".join(words))
        return reshaped_lines

    def create_image(self, text: str):
        try:
            font = ImageFont.truetype(self.font_name, self.font_size)
        except:
            font = ImageFont.load_default()
    
        # Calculate text size
        lines = self._preprocess_text(text)
        
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        print(f"line_height: {line_height}")
        
        image_height = line_height * len(lines) + 2*self.padding
        # Calculate image width more accurately
        image_width = max(font.getbbox(line)[2] for line in lines) + 2 * self.padding
        # Create an image
        img = Image.new("RGB", (image_width, image_height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Draw the text
        y_text = self.padding
        for line in lines:
            draw.text((self.padding, y_text), line, self.text_color, font=font)
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
