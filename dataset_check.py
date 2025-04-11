from src.datasetWrapper import GSM8kWrapper, SQuADWrapper, TriviaQAWrapper


gsm8k_data = GSM8kWrapper()
with open("gsm8k_data.txt", "w") as gsm8k_file:
    gsm8k_file.write(str(gsm8k_data.dataset["question"]))

squad_data = SQuADWrapper()
with open("squad_data.txt", "w") as squad_file:
    squad_file.write(str(squad_data.dataset["question"]))

triviaqa_data = TriviaQAWrapper()
with open("triviaqa_data.txt", "w") as triviaqa_file:
    triviaqa_file.write(str(triviaqa_data.dataset["question"]))