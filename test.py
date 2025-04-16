from src.filters import *


import json


with open("parameters.json", "r") as f:
    parameters = json.load(f)


phrase_scared = parameters["saved text"]["scared"]

print(f"phrase: {phrase_scared}")
print("output:")
filter = PushFrontTextFilter(phrase_scared)
print(filter.apply_filter("hello"))