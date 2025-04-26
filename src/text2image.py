from PIL import Image, ImageDraw, ImageFont
from src.filters import AbstractTextFilter
from src.log import Log
import textwrap
import uuid

MAX_WORDS_PER_LINE__IMG = 10

class AbstractText2Image:
    def __init__(self):
        self.width = 800
        self.height = 300
        self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.text_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.padding = 10
        self.font_size = 20

    def create_image(self, text: str):
        return None

    def set_font_size(
        self,
        text: str,
        min_size: int = 5,
        max_size: int = 30,
        line_spacing: float = 1.15
    ) -> None:
        """
        Pick the largest font size so that, when wrapping on word boundaries
        to self.width, all lines (with line_spacing) fit within self.height.
        """
        pass
    
class FixedSizeText2Image(AbstractText2Image):
    def __init__(self, width: int = 1000, height: int = 600, font_path='DejaVuSans.ttf' , text_color=(0, 0, 0), bg_color=(255, 255, 255), padding: int = 15, default_font_size: int = 20, longest_text=None):
        super().__init__()
        self.width = width
        self.height = height
        self.font_path = font_path
        self.text_color = text_color
        self.bg_color = bg_color
        self.padding = padding
        self.font_size = self.set_font_size(longest_text) if longest_text else default_font_size


    def set_font_size(
        self,
        text: str,
        min_size: int = 5,
        max_size: int = 60,
        line_spacing: float = 1.15
    ) -> None:
        """
        Pick the largest font size so that, when wrapping on word boundaries
        to self.width, all lines (with line_spacing) fit within self.height.
        """
        max_size = max_size
        W = self.width - 2 * self.padding
        H = self.height - 2 * self.padding
        lo, hi = min_size, max_size
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(self.font_path, mid)

            # wrap text by checking each word’s bbox
            draw = ImageDraw.Draw(Image.new("RGB",(1,1)))
            words = text.split()
            lines = []
            current = ""
            for w in words:
                test = (current + " " + w).strip()
                x0, y0, x1, y1 = draw.textbbox((0,0), test, font=font)
                if (x1 - x0) <= W:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = w
            if current:
                lines.append(current)

            # compute total height with your line spacing
            ascent, descent = font.getmetrics()
            line_height  = (ascent + descent) * line_spacing
            total_height = len(lines) * line_height

            # include padding
            if total_height< H:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        self.font_size = best
    
    def create_image(self, text):                  
        W, H = self.width, self.height
        padding = self.padding
        usable_width = W - 2 * padding

        font = ImageFont.truetype(self.font_path, self.font_size)

        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        lines = []
        draw_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        words = text.split()
        current_line = ""
        for word in words:
            test_line = (current_line + " " + word).strip()
            x0, y0, x1, y1 = draw_dummy.textbbox((0, 0), test_line, font=font)
            if (x1 - x0) <= usable_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * 1.15)

        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            y = padding + i * line_height
            draw.text((padding, y), line, fill="black", font=font)

        #filename = f"new_text_image_{uuid.uuid4().hex[:8]}.png"
        #img.save(filename)
        #Log().logger.info(f"Image saved as: {filename}")


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



