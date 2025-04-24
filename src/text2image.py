from PIL import Image, ImageDraw, ImageFont
from src.filters import AbstractTextFilter
from src.log import Log
import textwrap
import uuid

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

    def find_font(self, text: str):
        
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

    def create_image(
        self,
        text,                    
        image_size = (800, 300),             
        font_size = 14,               
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",                    
    ):                           
        W, H = image_size

        font_size = self.find_font(text) if font_size is None else font_size
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        avg_char_width = sum(draw.textsize(c, font=font)[0] for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") / 52
        chars_per_line = max(1, int(W / avg_char_width))
        lines = textwrap.wrap(text, width=chars_per_line)

        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * 1.15)

        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            y = i * line_height
            draw.text((0, y), line, fill="black", font=font)
        #plot the image 

        filename = f"text_image_{uuid.uuid4().hex[:8]}.png"
        img.save(filename)
        print(f"Image saved as: {filename}")


        return img

        
    
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