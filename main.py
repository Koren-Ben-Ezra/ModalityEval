from src.eval import *

def main():
    with_and_without_cot_instruction_eval()
    flip2LettersTextFilter_TF_eval()
    flip2LettersTextFilter_IF_eval()
    shuffle_p_increase_image_eval()
    shuffle_p_increase_text_eval()
    basic_eval_all()
    personalized_information_text_eval()
    personalized_information_image_eval()

if __name__ == "__main__":
    main()
