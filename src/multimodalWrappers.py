from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt
import torch
import re

TXT_INSTRUCTION = "[INSTRUCTION] Provide only the final answer in plain text. No explanations or additional words. For example, if the answer is 5, respond with $5$ only."
IMG_INSTRUCTION = "[INSTRUCTION] The question is displayed within the provided image. Respond solely with the final answer in text form. No explanations or extra words. For example, if the answer is 5, respond with $5$ only."
MAX_NEW_TOKENS = 30


def extract_number(text: str) -> str:
    """
    Cleans the text by removing special tokens and then extracts the last number found.
    Returns the number as a float. If no number is found, returns None.
    """
    # Remove special tokens like <|begin_of_text|>, <|end_header_id|>, etc.
    cleaned_text = re.sub(r'<\|.*?\|>', '', text)
    # Remove extra whitespace
    cleaned_text = cleaned_text.strip()
    # Find all numbers (including decimals)
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_text)
    if matches:
        return str(matches[-1])
    return ""

class MultimodalWrapper:
    def __init__(self):
        self.model_id: str = None
    
    def generate_ans_from_image(self, img: Image):
        return None
    
    def generate_ans_from_text(self, text: str):
        return None


# class ModelManager:
#     def __init__(self, multimodal: MultimodalWrapper):
#         self._multimodal = multimodal
#         self.model_name = self._multimodal.model_name

#     def separate_forward(self, text_q: str, img_q: Image):
#         answer_text_q = self._multimodal.generate_ans_from_text(text_q)
#         answer_img_q = self._multimodal.generate_ans_from_image(img_q)
#         return answer_text_q, answer_img_q
    
    


class LlamaWrapper(MultimodalWrapper):
    def __init__(self, model_id: str="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.model_id = model_id
        self.model_name= model_id.split('/')[-1]
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.tie_weights() 
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        
    def generate_ans_from_image(self, image: Image):
        image = image.convert("RGB")  # keep it 3-channel

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": IMG_INSTRUCTION}
            ]}
        ]
        input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self._model.device)

        output = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded_output = self._processor.decode(output[0])
        return extract_number(decoded_output)


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
        final_output = extract_number(decoded_output)
        return final_output

    
    # def _postprocess_output(self, output):
    #     return output.split("####")[1].strip()