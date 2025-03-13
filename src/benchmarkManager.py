from typing import List
import os

import pandas as pd
from tqdm import tqdm

from src.filters import AbstractImageFilter, AbstractTextFilter
from src.category import Category
from src.datasetWrapper import AbstractDatasetWrapper
from src.multimodalWrappers import ModelManager

BENCHMARKS_DIR = "benchmarks"

class BenchmarkManager:
    def __init__(self, datasetWrapper: AbstractDatasetWrapper, model_manager: ModelManager, metadata: dict, save_predictions: bool=False):
        self._datasetWrapper = datasetWrapper
        self._model_manager = model_manager
        self.metadata = metadata
        self.save_predictions = save_predictions
        
        self._categories: List[Category] = []
        self.benchmark_name = BENCHMARKS_DIR + "\\" + "_".join(self.metadata.values())
        self._make_benchmark_dir()
        
    def execute_test(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter):
        category = Category(text_f, img_f, self.benchmark_name, self.save_predictions)
        self._categories.append(category)
        
        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category)
            
        category.save_predictions()
            
    def _execute_single_prompt(self, sample, category: Category):
        
        filtered_text = category.text_f.apply_filter(sample["question"])
        filtered_image = category.img_f.apply_filter(sample["question_image"])
        
        prediction_from_text, prediction_from_img = self._model_manager.separate_forward(filtered_text, filtered_image)
        
        self._update_category_stats(sample, category, prediction_from_text, prediction_from_img)
    
    def _update_category_stats(self, sample, category: Category, prediction_from_text: str, prediction_from_img: str):
        
        category.text_stats.total += 1
        category.img_stats.total += 1
        answer = sample["answer"]
        
        if prediction_from_text == answer:
            category.text_stats.success += 1
        
        if prediction_from_img == answer:
            category.img_stats.success += 1
        
        if category.predictions_df is not None:
            new_row = pd.DataFrame([{ 
                 "answer": sample["answer"], 
                 "text_prediction": prediction_from_text,
                 "img_prediction": prediction_from_img}])
            
            category.predictions_df = pd.concat([category.predictions_df, new_row])
    
    def _make_benchmark_dir(self):
        if not os.path.exists(BENCHMARKS_DIR):
            os.makedirs(BENCHMARKS_DIR)
        
        if not os.path.exists(self.benchmark_name):
            os.makedirs(self.benchmark_name)
    
    def _create_summary(self):
        summary_df = pd.DataFrame(columns=["Text Filter", "Image Filter", "Text Accuracy", "Image Accuracy"])
        for category in self._categories:
            text_acc, img_acc = category.eval_accuracy()
            new_row = pd.DataFrame([{"Text Filter": category.text_f.filter_name,
                                        "Image Filter": category.img_f.filter_name,
                                        "Text Accuracy": text_acc,
                                        "Image Accuracy": img_acc}])
            summary_df = pd.concat([summary_df, new_row])
        
        return summary_df
    
    def write_summary(self):
        # Save to CSV with metadata as a header
        filename = self.benchmark_name + "\\" + "summary.csv"
        with open(filename, "w") as file:
            for key, value in self.metadata.items():
                file.write(f"# {key}: {value}\n")  # Writing metadata as comments
            self._create_summary().to_csv(file, index=False)
