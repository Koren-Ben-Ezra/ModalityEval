import os

import matplotlib.pyplot as plt

from src.filters import *
from src.text2image import *

TEST_DIR = "test_output"

def clear_test_dir():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    for filename in os.listdir(TEST_DIR):
        os.remove(os.path.join(TEST_DIR, filename))


def test_text_filter(filter: AbstractTextFilter, input: str, answer: str):
    try:
        output = filter.apply_filter(input, answer)
    except Exception as e:
        print(f"\n[Error] {filter.filter_name}: {e}")
        return
    
    summary = f"{filter.filter_name}\n\nInput:\n{input}\n\nOutput:\n{output}" 

    with open(os.path.join(TEST_DIR, f"{filter.filter_name}.txt"), "w") as f:
        f.write(summary)


def test_image_filter(filter: AbstractImageFilter, input: Image, answer: str):
    
    try:
        output = filter.apply_filter(input, answer)
    except Exception as e:
        print(f"\n[Error] {filter.filter_name}: {e}")
        return

    plt.subplot(1, 2, 1)
    plt.suptitle(f"{filter.filter_name}")
    plt.imshow(input)
    plt.title("input")
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("output")
    plt.savefig(os.path.join(TEST_DIR, f"{filter.filter_name}.png"))


# Setup test environment
clear_test_dir()
text2image = FixedSizeText2Image()
# text2image = FilteredFixedSizeText2Image(ShuffleWordTextFilter())
text_input = """Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?"""
answer = "18"
image_input = text2image.create_image(text_input)


# Test identity filters
# test_text_filter(IdentityTextFilter(), text_input, answer)
# test_image_filter(IdentityImageFilter(), image_input, answer)


# Test noise filters
# test_image_filter(ContrastStretchingImageFilter(), image_input, answer)
# test_image_filter(HistogramEqualizationImageFilter(), image_input, answer)
# test_image_filter(GaussianImageFilter(), image_input, answer)
# test_text_filter(ShuffleWordTextFilter(), text_input, answer)
# test_text_filter(SwapWordsTextFilter(), text_input, answer)


# Test general information filters
# img = Image.open("my_image.jpg")
# test_text_filter(ReplaceBackgroundTextFilter(img), text_input, answer)
# test_text_filter(PushFrontTextFilter(), text_input, answer)
# test_text_filter(PushTopImageFilter(), text_input, answer)

# Test personal information filters