# from openai_key import API_KEY
# from openai import OpenAI 
# import matplotlib.pyplot as plt
# import re
# from src.text2image import FixedSizeText2Image
# import base64
# import io

# ## Set the API key and model name
# MODEL="gpt-4o-mini"
# client = OpenAI(api_key=API_KEY)

# def extract_number(text: str) -> str:
#     """
#     Removes special tokens and extracts the last numeric value found (including decimals).
#     Returns the number as a string. If no number is found, returns an empty string.
#     """
#     cleaned_text = re.sub(r'<\|.*?\|>', '', text).strip()
#     matches = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_text)
#     if matches:
#         return matches[-1]  # Return the last match
#     return ""

# text2image = FixedSizeText2Image()
# # text2image = FilteredFixedSizeText2Image(ShuffleWordTextFilter())
# text_input = """Janet's ducks lay 16 eggs per day. She eats three for breakfast 
# every morning and bakes muffins for her friends every day with four. 
# She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
# How much in dollars does she make every day at the farmers' market?"""
# correct_answer = "18"
# image_input = text2image.create_image(text_input)



# completion_text = client.chat.completions.create(
#   model=MODEL,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
#     {"role": "user", "content": text_input}  # <-- This is the user message for which the model will generate a response
#   ]
# )


# #base64_image = encode_image(IMAGE_PATH)
# buffer = io.BytesIO()
# image_input.save(buffer, format="PNG")
# buffer.seek(0)  # Rewind to the start
# base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
# completion_image = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
#         {"role": "user", "content": [
#             {"type": "text", "text": ""},
#             {"type": "image_url", "image_url": {
#                 "url": f"data:image/png;base64,{base64_image}"}
#             }
#         ]}
#     ],
#     temperature=0.0,
# )



# # Extract the raw text from the completion
# text_only_response = completion_text.choices[0].message.content
# print("[Text Only] Raw Model Answer:", text_only_response)
# text_only_extracted = extract_number(text_only_response)
# print("[Text Only] Extracted Answer:", text_only_extracted)
# if text_only_extracted == correct_answer:
#     print("[Text Only] Model answer matches the correct answer.")
# else:
#     print("[Text Only] Model answer does NOT match the correct answer.")



# # Extract the raw text from the completion
# image_response = completion_image.choices[0].message.content
# print("\n[Image Input] Raw Model Answer:", image_response)
# image_extracted = extract_number(image_response)
# print("[Image Input] Extracted Answer:", image_extracted)
# if image_extracted == correct_answer:
#     print("[Image Input] Model answer matches the correct answer.")
# else:
#     print("[Image Input] Model answer does NOT match the correct answer.")


