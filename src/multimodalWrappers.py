from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import re
import json

from src.log import Log

PARAMETERS_PATH = "parameters.json"

with open(PARAMETERS_PATH, 'r') as file:
    data = json.load(file)
    system = data.get('system', '')

# TXT_INSTRUCTION = system.get('text instruction', '')
# IMG_INSTRUCTION = system.get('image instruction', '')

TXT_INSTRUCTION = system.get('CoT text instruction', '')
IMG_INSTRUCTION = system.get('CoT image instruction', '')

MAX_NEW_TOKENS = 1000

#for "gpt-4o-mini"

def extract_number(text: str) -> str:
    """
    Removes special tokens and extracts the last numeric value found (including decimals).
    Returns the number as a string. If no number is found, returns an empty string.
    """
    cleaned_text = re.sub(r'<\|.*?\|>', '', text).strip()
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_text)
    if matches:
        return matches[-1]  # Return the last match
    return ""

class MultimodalWrapper:
    def __init__(self):
        self.model_id: str = None
        self.model_name: str = None
    
    def generate_ans_from_image(self, img: Image):
        return None
    
    def generate_ans_from_text(self, text: str):
        return None

class LlamaWrapper(MultimodalWrapper):
    def __init__(self, model_id: str="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        Log().logger.info("Loading Llama model...")
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        try:
            self._model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            Log().logger.error(f"Error loading model: {e}")
            raise e
            
        try:
            self._model.tie_weights()
        except Exception as e:
            Log().logger.error(f"Error tying weights: {e}")
            raise e

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            Log().logger.error(f"Error loading processor: {e}")
            raise e
        
        Log().logger.info(f"Model {self.model_name} loaded successfully.")
        
        if torch.cuda.is_available():
            Log().logger.info(f"GPU available: {torch.cuda.get_device_name(0)}, device count: {torch.cuda.device_count()}")
        else:
            Log().logger.warning("No GPU available.")
            
        Log().logger.info(f"Model is running on {self._model.device}.")            
        
        
    def generate_ans_from_image(self, image: Image):
        image = image.convert("RGB")  # keep it 3-channel

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": IMG_INSTRUCTION}
            ]}
        ]
        # input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        input_text = self._processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self._processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self._model.device)

        output = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded_output = self._processor.decode(output[0])
        
        try:
            output = self.extract_answer(decoded_output)
        except Exception as e:
            Log().logger.error(f"Error extracting answer: {e}")
            raise e
        return output
    
    def generate_ans_from_text(self, text: str):
        # Create a dummy 224x224 RGB image (NOT 1x1, not grayscale)
        image = Image.new(mode="RGB", size=(224, 224), color="white")
        
        txt_message = TXT_INSTRUCTION + f"\n[QUESTION] {text}"
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": txt_message}
            ]}
        ]
        input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        # Now pass this 224x224 RGB image (three channels) to the processor
        inputs = self._processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self._model.device)

        output = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded_output = self._processor.decode(output[0])
        
        try:
            output = self.extract_answer(decoded_output)
        except Exception as e:
            Log().logger.error(f"Error extracting answer: {e}")
            raise e
        
        return output
        
    def extract_answer(self, text: str, token: str="<|eot_id|>") -> str:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}: {text}")

        pattern = r"([-+]?\d+(?:\.\d+)?)(?:\s*" + re.escape(token) + ")"

        match = re.search(pattern, text)
        return match.group(1) if match else None

    

# class GPT4ominiWrapper(MultimodalWrapper):
#     client = OpenAI(api_key=API_KEY)

#     def __init__(self, model_id: str = "gpt-4o-mini"):
#         super().__init__()
#         self.model_id = model_id

#     # text is the question only without instructions
#     def generate_ans_from_text(self, text: str):
#         # Build the user prompt with the instruction
#         try:
#             completion_text = self.client.chat.completions.create(
#             model=self.model_id,
#             messages=[
#                 {"role": "system", "content": TXT_INSTRUCTION}, # <-- This is the system message that provides context to the model
#                 {"role": "user", "content": text}  # <-- This is the user message for which the model will generate a response
#             ]
#             )
#             text_only_response = completion_text.choices[0].message.content
#             # print("\n[Text Only] Raw Model Answer:", text_only_response)
#             text_only_extracted = extract_number(text_only_response)
#             # print("[Text Only] Extracted Answer:", text_only_extracted)
#             return text_only_extracted
#         except Exception as e:
#             print("Error querying GPT-4o with text input:", e)
#             return ""

#     def generate_ans_from_image(self, img: Image):
#         """
#         Hypothetical method: base64-encodes an image and sends it with an instruction
#         for GPT-4o to parse. Currently, OpenAI does NOT support images via the ChatCompletion API.
#         This code is for demonstration only and will not work with the real GPT-4 endpoint.
#         """
#         # Convert the PIL image to base64
#         buffer = io.BytesIO()
#         img.save(buffer, format="PNG")
#         buffer.seek(0)
#         base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

#         try:
#             completion_image = self.client.chat.completions.create(
#                 model=self.model_id,
#                 messages=[
#                     {"role": "system", "content": IMG_INSTRUCTION},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": ""},
#                         {"type": "image_url", "image_url": {
#                             "url": f"data:image/png;base64,{base64_img}"}
#                         }
#                     ]}
#                 ],
#                 temperature=0.0,
#             )
#             image_response = completion_image.choices[0].message.content
#             # print("\n[Image Input] Raw Model Answer:", image_response)
#             image_number_response = extract_number(image_response)
#             # print("[Image Input] Extracted Answer:", image_number_response)

#             # Save the image as a JPG in the "images" directory
#             # import os
#             # output_dir = "images"
#             # os.makedirs(output_dir, exist_ok=True)
#             # image_path = os.path.join(output_dir, "output_image.jpg")
#             # img.save(image_path, format="JPEG")
#             # print(f"Image saved to {image_path}")
            
#             return image_number_response
#         except Exception as e:
#             print("Error querying GPT-4o with image input (not supported):", e)
#             return ""
