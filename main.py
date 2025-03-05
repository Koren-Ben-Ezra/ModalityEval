# from benchmarkManager import BenchmarkManager
# import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image
from torchvision import transforms

def main():
    model_name = "meta-llama/Llama-Guard-3-11B-Vision"  # Replace with your model's repo name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # adjust size if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # common normalization for vision models
])

    image = Image.open("my_image.jpg")
    image_tensor = image_transform(image).unsqueeze(0).to("cuda")
    
    prompt = "The rickest rick"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Here, we assume the model accepts an argument for image inputs.
    # Check your model’s documentation for the exact argument name.
    outputs = model.generate(
        **inputs, 
        image_inputs=image_tensor,  # This argument may differ
        max_new_tokens=100
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()