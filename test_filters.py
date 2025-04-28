import os

import matplotlib.pyplot as plt
from PIL import Image

from src.filters import *
from src.text2image import *
import json

TEST_DIR = "test_filters_output"

def clear_test_dir():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    for filename in os.listdir(TEST_DIR):
        os.remove(os.path.join(TEST_DIR, filename))


def test_text_filter(filter: AbstractTextFilter, input: str, answer: str, title: str = ""):
    try:
        output = filter.apply_filter(input, answer)
    except Exception as e:
        print(f"\n[Error] {filter.filter_name}: {e}")
        return
    
    summary = f"{filter.filter_name}\n\nInput:\n{input}\n\nOutput:\n{output}"
    name = title if title else filter.filter_name
    with open(os.path.join(TEST_DIR, f"{name}.txt"), "w") as f:
        f.write(summary)


def test_image_filter(filter: AbstractImageFilter, input: Image, answer: str, title: str = ""):
    
    try:
        output = filter.apply_filter(input, answer)
    except Exception as e:
        print(f"\n[Error] {filter.filter_name}: {e}")
        return
    name = title if title else filter.filter_name
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([]) 
    plt.suptitle(filter.filter_name)
    plt.imshow(input)
    plt.title("input")
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([]) 
    plt.imshow(output)
    plt.title("output")
    plt.savefig(os.path.join(TEST_DIR, f"{name}.png"))


# Setup test environment
clear_test_dir()

text2image = FixedSizeText2Image(font_path="DejaVuSans.ttf")
# text2image = FilteredFixedSizeText2Image(ShuffleWordTextFilter())
text_input = """Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?"""
answer = "18"
image_input = text2image.create_image(text_input)


# # Test identity filters
test_text_filter(IdentityTextFilter(), text_input, answer)
test_image_filter(IdentityImageFilter(), image_input, answer)


# # Test noise filters
# test_image_filter(HistogramEqualizationImageFilter(), image_input, answer)
# test_image_filter(GaussianImageFilter(), image_input, answer)
# test_text_filter(ShuffleWordTextFilter(), text_input, answer)
# test_text_filter(SwapWordsTextFilter(), text_input, answer)


# # Test general information filters
# img = Image.open("images/amanda.jpg")
# phrase = "My name is John Doe. I live in New York City. I am a software engineer at Google."
# test_image_filter(ReplaceBackgroundImageFilter(img), image_input, answer)
# test_text_filter(PushFrontTextFilter(phrase), text_input, answer)
# test_image_filter(PushTopImageFilter(img), image_input, answer)
# # Test personal information filters
# test_image_filter(SurroundByCorrectAnsImageFilter(), image_input, answer)
# test_image_filter(SurroundByWrongAnsImageFilter(), image_input, answer)
# test_image_filter(SurroundByPartialCorrectAnsImageFilter(), image_input, answer)

# test_text_filter(SurroundByCorrectAnsTextFilter(), text_input, answer)
# test_text_filter(SurroundByWrongAnsTextFilter(), text_input, answer)
# test_text_filter(SurroundByPartialCorrectAnsTextFilter(), text_input, answer)


# phrases images:
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    phrases = parameters["phrases"]

# text2image.height *= 5
# text2image.width *= 5
for key, phrase in phrases.items():
    question = PushFrontTextFilter(phrase).apply_filter(text_input)
    text2image.set_font_size(question)
    
    img = text2image.create_image(question)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(key)
    plt.savefig(os.path.join(TEST_DIR, f"{key}.png"))
    plt.close()
    