from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch


class MultimodalWrapper:
    def generate_ans_from_image(self, img: Image):
        return None
    
    def generate_ans_from_text(self, text: str):
        return None


class ModelManager:
    def __init__(self, multimodal: MultimodalWrapper):
        self._multimodal = multimodal

    def separate_forward(self, text_q: str, img_q: Image):
        answer_text_q = self._multimodal.generate_ans_from_text(text_q)
        answer_img_q = self._multimodal.generate_ans_from_image(img_q)
        return answer_text_q, answer_img_q


class LlamaWrapper(MultimodalWrapper):
    def __init__(self, model_id: str="meta-llama/Llama-3.2-11B-Vision"):
        self._model_id = model_id
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        
    def generate_ans_from_image(self, img: Image):
        prompt = "<|image|><|begin_of_text|>"
        inputs = self._processor(img, prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=30)
        return output
    
    def generate_ans_from_text(self, text: str):
        img = Image.new(mode="RGB", size=(2, 2), color="white")
        text = "<|image|><|begin_of_text|>" + text
        inputs = self._processor(img, text, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=30)
        return output

