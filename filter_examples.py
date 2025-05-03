import os

import matplotlib.pyplot as plt
from PIL import Image

from src.filters import *
from src.text2image import *
import json

EXAMPLE_DIR = "filter_examples"

def clear_test_dir():
    if not os.path.exists(EXAMPLE_DIR):
        os.makedirs(EXAMPLE_DIR)
    
    for filename in os.listdir(EXAMPLE_DIR):
        os.remove(os.path.join(EXAMPLE_DIR, filename))

clear_test_dir()

text2image = FixedSizeText2Image(font_path="DejaVuSans.ttf")
text_input = """Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?"""
answer = "18"

# text2image.height = 300
text2image.set_font_size(text_input, max_size=35)


def generate_image(text_input, filename):
    img = text2image.create_image(text_input)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(EXAMPLE_DIR, filename), bbox_inches='tight', pad_inches=0)
    plt.close()
    return img
    
base_img = generate_image(text_input, "identity.png")

shuffle_text = ShuffleWordTextFilter().apply_filter(text_input, answer)
generate_image(shuffle_text, "shuffle.png")

flip2_text = flip2LettersTextFilter().apply_filter(text_input, answer)
generate_image(flip2_text, "flip2.png")

correct_img = SurroundByCorrectAnsImageFilter().apply_filter(base_img, answer)
partial_img = SurroundByPartialCorrectAnsImageFilter().apply_filter(base_img, answer)
wrong_img =   SurroundByWrongAnsImageFilter().apply_filter(base_img, answer)


with open("parameters.json", 'r') as file:
    data = json.load(file)
    phrases = data["phrases"]

stress_short_phrase = phrases["military_short"] 
stress_text = PushFrontTextFilter(stress_short_phrase).apply_filter(text_input, answer)
text2image.set_font_size(stress_text)
generate_image(stress_text, "stress.png")

relax_short_phrase = phrases["relax_short"]
relax_text = PushFrontTextFilter(relax_short_phrase).apply_filter(text_input, answer)
text2image.set_font_size(relax_text)
generate_image(relax_text, "relax.png")