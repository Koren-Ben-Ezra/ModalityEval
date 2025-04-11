from src.datasetWrapper import GSM8kWrapper, SQuADWrapper, TriviaQAWrapper


gsm8k_data = GSM8kWrapper()
with open("gsm8k_data.txt", "w") as gsm8k_file:
    for sample in gsm8k_data.dataset:
        # Assuming you want to write the question and answer to the file
        gsm8k_file.write(f"Question: {sample['question']}\n")
        gsm8k_file.write(f"Answer: {sample['answer']}\n")
        gsm8k_file.write("\n")

squad_data = SQuADWrapper()
with open("squad_data.txt", "w") as squad_file:
    for sample in squad_data.dataset:
        # Assuming you want to write the question and answer to the file
        squad_file.write(f"Question: {sample['question']}\n")
        squad_file.write(f"Answer: {sample['answer']}\n")
        squad_file.write("\n")

triviaqa_data = TriviaQAWrapper()
with open("triviaqa_data.txt", "w") as triviaqa_file:
    for sample in triviaqa_data.dataset:
        # Assuming you want to write the question and answer to the file
        triviaqa_file.write(f"Question: {sample['question']}\n")
        triviaqa_file.write(f"Answer: {sample['answer']}\n")
        triviaqa_file.write("\n")