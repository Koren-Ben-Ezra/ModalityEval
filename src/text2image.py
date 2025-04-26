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
    

    
class FixedSizeText2Image(AbstractText2Image):
    """
    Uses a fixed image size (width, height). Automatically adjusts the font size
    so the text fits. The text is then centered within the final image, taking
    bounding box offsets and padding into account to avoid clipping.
    """

    def __init__(self, width: int = 800, height: int = 300, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', text_color=(0, 0, 0), bg_color=(255, 255, 255), padding: int = 10,font_size: int = 20, longest_text=None):
        super().__init__()
        self.width = width
        self.height = height
        self.font_path = font_path
        self.text_color = text_color
        self.bg_color = bg_color
        self.padding = padding
        self.font_size = self.find_font_size(longest_text) if longest_text else font_size



    def find_font_size(
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


        

class FixedFontText2Image(AbstractText2Image):

    def __init__(self):
        super().__init__()

    def create_image(self, text: str):
        text = self.preprocess_text(text)
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

    def preprocess_text(self,text: str):
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



