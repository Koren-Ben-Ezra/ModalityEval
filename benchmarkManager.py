from typing import List
from tqdm import tqdm

from filters import AbstractImageFilter, AbstractTextFilter, Category
from datasetWrapper import AbstractDatasetWrapper
from multimodalWrappers import ModelManager

class BenchmarkManager:
    def __init__(self, datasetWrapper: AbstractDatasetWrapper, model_manager: ModelManager):
        self._datasetWrapper = datasetWrapper
        self._model_manager = model_manager
        self._categories: List[Category] = []
        
    def execute_test(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter):
        category = Category(text_f, img_f)
        self._categories.append(category)
        
        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category)
            
        return category.eval()
    
    def _execute_single_prompt(self, sample, category: Category):
        
        filtered_text = category.text_f.apply_filter(sample["question"])
        filtered_image = category.img_f.apply_filter(sample["question_image"])
        
        answers = self._model_manager.separate_forward(filtered_text, filtered_image)
        true_answer = sample['answer']
        
        if answers[0] == true_answer:
            category.text_stats += 1
        if answers[1] == true_answer:
            category.img_stats += 1
