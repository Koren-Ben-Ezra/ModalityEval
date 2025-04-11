import re
import base64
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as IPythonImage, display

# Hypothetical imports for an OpenAI-like client
# Adjust as needed for your environment
from openai_key import API_KEY
from openai import OpenAI  # Fictional 'openai' client that supports image_url param
                           # Real openai.ChatCompletion doesn't currently accept images.

# 1) The first question from GSM8K (example)
GSM8K_Q1 = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# 2) Create a plain white image with the question text in the middle
def create_question_image(text, filename="gsm8k_q1.png", width=800, height=400):
    # Create a white background
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try loading a default font. On some systems, you may need a TTF path.
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        # Fall back to a basic PIL font if arial.ttf isn't found
        font = ImageFont.load_default()

    # Word-wrap logic (optional)
    lines = []
    line_width = 0
    max_width = width - 40  # some padding
    for word in text.split():
        test_line = (lines[-1] + " " + word) if lines else word
        w, _ = draw.textsize(test_line, font=font)
        if w <= max_width:
            if lines:
                lines[-1] = test_line
            else:
                lines.append(test_line)
        else:
            lines.append(word)

    # Calculate total text height
    line_height = draw.textsize("Test", font=font)[1]
    total_text_height = line_height * len(lines)

    # Compute vertical start so the text is centered
    y_start = (height - total_text_height) // 2

    # Draw each line centered horizontally
    for line in lines:
        w, _ = draw.textsize(line, font=font)
        x_pos = (width - w) // 2
        draw.text((x_pos, y_start), line, fill="black", font=font)
        y_start += line_height

    img.save(filename)

# 3) Encode the image in Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 4) Regex-based numeric extractor
def extract_number(text: str) -> str:
    """
    Cleans the text by removing special tokens and then extracts the last number found.
    Returns the number as a string. If no number is found, returns an empty string.
    """
    cleaned_text = re.sub(r'<\|.*?\|>', '', text).strip()
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_text)
    if matches:
        return matches[-1]
    return ""

#
# MAIN DEMO
#

# A) Create the white image with the GSM8K question
create_question_image(GSM8K_Q1, filename="gsm8k_q1.png")
# Preview the image inline (Jupyter/IPython only)
display(IPythonImage("gsm8k_q1.png"))

# B) Prepare to send it to the hypothetical GPT-4o endpoint
base64_image = encode_image("gsm8k_q1.png")

# Example client config
MODEL = "gpt-4o-mini"  # Hypothetical model name
client = OpenAI(api_key=API_KEY)

# C) Create a system + user message
# For demonstration, we embed the base64 image data in the "image_url" field
messages = [
    {"role": "system", "content": "You are a helpful math assistant. Please answer only with the numeric result."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is a math question in the attached image. Please provide only the numeric answer."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    }
]

# D) Make the (hypothetical) API call
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.0
)

raw_answer = response.choices[0].message.content
print("Raw model answer:", raw_answer)

# E) Extract the numeric answer
numeric_answer = extract_number(raw_answer)
print("Extracted numeric answer:", numeric_answer)
