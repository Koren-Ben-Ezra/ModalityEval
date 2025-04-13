from PIL import Image, ImageDraw, ImageFont
from src.filters import AbstractTextFilter
from src.log import Log

MAX_WORDS_PER_LINE__IMG = 10

class AbstractText2Image:
    def create_image(self, text: str):
        return None
    
    @staticmethod
    def _preprocess_text(text: str):
        words = []
        for line in text.split("\n"):
            for word in line.split(" "):
                words.append(word)
        res = []
        new_line = []
        for word in words:
            if len(new_line) == MAX_WORDS_PER_LINE__IMG:
                res.append(" ".join(new_line))
                new_line = []
            
            new_line.append(word)
        
        return "\n".join(res)

class FixedFontText2Image(AbstractText2Image):
    """
    Uses a fixed font size. The resulting image is sized exactly
    to the text's bounding box and accounts for any offsets
    so the text isn't clipped. Also supports optional padding.
    """
    def __init__(
        self,
        font_path="arial.ttf",
        font_size=20,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        padding=0
    ):
        """
        :param font_path: Path to the .ttf or .otf font file.
        :param font_size: The fixed font size.
        :param text_color: (R, G, B) tuple for text color.
        :param bg_color: (R, G, B) tuple for background color.
        :param padding: Integer number of pixels to pad on all sides.
        """
        self.font_path = font_path
        self.font_size = font_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.padding = padding

    def create_image(self, text: str):
        text = AbstractText2Image._preprocess_text(text)
        # 1. Create a small image to measure the bounding box of the text
        temp_img = Image.new("RGB", (1, 1), self.bg_color)
        temp_draw = ImageDraw.Draw(temp_img)
        font = ImageFont.truetype(self.font_path, self.font_size)

        # textbbox returns (left, top, right, bottom)
        left, top, right, bottom = temp_draw.textbbox((0, 0), text, font=font)

        text_width = right - left
        text_height = bottom - top

        # 2. Create the final image sized to the text plus padding
        width = text_width + 2 * self.padding
        height = text_height + 2 * self.padding
        image = Image.new("RGB", (width, height), self.bg_color)
        draw = ImageDraw.Draw(image)

        # 3. Draw text at a negative offset, plus the padding
        #    This ensures the text isn't clipped and is padded
        draw.text(
            (self.padding - left, self.padding - top),
            text,
            font=font,
            fill=self.text_color
        )

        return image


class FixedSizeText2Image(AbstractText2Image):
    """
    Uses a fixed image size (width, height). Automatically adjusts the font size
    so the text fits. The text is then centered within the final image, taking
    bounding box offsets and padding into account to avoid clipping.
    """
    def __init__(
        self,
        width=800,
        height=300,
        font_path="arial.ttf",
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        max_font_size=100,
        padding=20
    ):
        self.width = width
        self.height = height
        self.font_path = font_path
        self.text_color = text_color
        self.bg_color = bg_color
        self.max_font_size = max_font_size
        self.padding = padding

    def create_image(self, text: str):
        
        try:
            text = AbstractText2Image._preprocess_text(text)
        except Exception as e:
            Log().logger.error(f"Error in preprocessing text: {e}")
            raise e
        
        chosen_font_size = 1

        # Effective drawing area (subtract padding from each side)
        effective_width = self.width - 2 * self.padding
        effective_height = self.height - 2 * self.padding

        # 1. Find the largest font size that fits the given dimensions minus padding
        for size in range(self.max_font_size, 0, -1):
            temp_img = Image.new("RGB", (1, 1), self.bg_color)
            temp_draw = ImageDraw.Draw(temp_img)
            font = ImageFont.truetype(self.font_path, size)
            left, top, right, bottom = temp_draw.textbbox((0, 0), text, font=font)

            text_width = right - left
            text_height = bottom - top

            if text_width <= effective_width and text_height <= effective_height:
                chosen_font_size = size
                break

        # 2. Create the final image
        image = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(image)
        final_font = ImageFont.truetype(self.font_path, chosen_font_size)

        # 3. Measure text with the chosen font, then center it in the padded area
        left, top, right, bottom = draw.textbbox((0, 0), text, font=final_font)
        text_width = right - left
        text_height = bottom - top

        # Center the text within the effective (padded) area
        x_offset = self.padding + (effective_width - text_width) // 2 - left
        y_offset = self.padding + (effective_height - text_height) // 2 - top

        draw.text((x_offset, y_offset), text, font=final_font, fill=self.text_color)
        return image
    
class FilteredFixedSizeText2Image(FixedSizeText2Image):
    def __init__(self, filter: AbstractTextFilter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter = filter
    
    def create_image(self, text: str):
        try:
            text = self.filter.apply_filter(text)
        except Exception as e:
            Log().logger.error(f"Error in applying filter: {e}")
            raise e
        return super().create_image(text)