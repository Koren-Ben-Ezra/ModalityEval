from typing import List 
import os
import re
import pandas as pd
from tqdm import tqdm

from src.filters import AbstractImageFilter, AbstractTextFilter
from src.category import Category
from src.datasetWrapper import AbstractDatasetWrapper
from src.multimodalWrappers import ModelManager

BENCHMARKS_DIR = "benchmarks"
EPSILON = 1e-3

class BenchmarkManager:
    def __init__(self, datasetWrapper: AbstractDatasetWrapper, model_manager: ModelManager, metadata: dict, save_predictions: bool=False):
        self._datasetWrapper = datasetWrapper
        self._model_manager = model_manager
        self.metadata = metadata
        self.save_predictions = save_predictions
        
        self._categories: List[Category] = []

        self.benchmark_name = os.path.join(
            BENCHMARKS_DIR, 
            self.metadata["Model"], 
            self.metadata["Dataset"]
        )
        self._make_benchmark_dir()
        print(f"[BenchmarkManager __init__] Benchmark directory: {self.benchmark_name}", flush=True)

        
    def execute_test(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter):
        category = Category(text_f, img_f, self.benchmark_name, self.save_predictions)
        self._categories.append(category)
        print("[execute_test] Starting test execution...", flush=True)
        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category)
        category.save_predictions()
        print("[execute_test] Test execution completed.", flush=True)
            
    def _execute_single_prompt(self, sample, category: Category):
        filtered_text = category.text_f.apply_filter(sample["question"])
        filtered_image = category.img_f.apply_filter(sample["question_image"])
        print("Filtered text:", filtered_text, flush=True)
        # If filtered_image is transformed to a specific format (or shape):
        print("Filtered image shape:", filtered_image.size, flush=True)
        prediction_from_text, prediction_from_img = self._model_manager.separate_forward(filtered_text, filtered_image)
        print("Prediction from text:", prediction_from_text, flush=True)
        print("Prediction from image:", prediction_from_img, flush=True)
        self._update_category_stats(sample, category, prediction_from_text, prediction_from_img)
    
    def _update_category_stats(self, sample, category: Category, pred_from_text: str, pred_from_img: str):
        print("[_update_category_stats] Updating stats...", flush=True)
        category.text_stats.total += 1
        category.img_stats.total += 1
        answer = sample["answer"]
        print(f"Question: {sample['question']}", flush=True)
        print(f"Answer (Ground Truth): {answer}", flush=True)
        print(f"Text Prediction: {pred_from_text}", flush=True)
        print(f"Image Prediction: {pred_from_img}", flush=True)
        if pred_from_text == answer:
            category.text_stats.success += 1
        
        print("Text match:", pred_from_text == answer, flush=True)
        if pred_from_img == answer:
            category.img_stats.success += 1
        print("Image match:", pred_from_img == answer, flush=True)

        # Append predictions if predictions_df exists
        if category.predictions_df is not None:
            new_row = pd.DataFrame([{
                "answer": answer,
                "text_prediction": pred_from_text,
                "img_prediction": pred_from_img
            }])
            category.predictions_df = pd.concat([category.predictions_df, new_row],ignore_index=True)
            print(f"[_update_category_stats] Appended new row. Total rows now: {len(category.predictions_df)}", flush=True)
    
    def _make_benchmark_dir(self):
        if not os.path.exists(BENCHMARKS_DIR):
            os.makedirs(BENCHMARKS_DIR)
        if not os.path.exists(self.benchmark_name):
            os.makedirs(self.benchmark_name)
    
    def _create_summary(self):
        try:
            summary_df = pd.DataFrame(columns=["Text Filter", "Image Filter", "Total Samples", "Text Correct", "Image Correct"])
            for category in self._categories:
                #text_acc, img_acc = category.eval_accuracy()
                new_row = pd.DataFrame([{
                    "Text Filter": category.text_f.filter_name,
                    "Image Filter": category.img_f.filter_name,
                    "Total Samples": category.text_stats.total,
                    "Text Correct": category.text_stats.success,
                    "Image Correct": category.img_stats.success
                }])
                try:
                    summary_df = pd.concat([summary_df, new_row], ignore_index=True)
                except Exception as e:
                    print("[_create_summary] Error during concatenation:", e, flush=True)
            return summary_df
        except Exception as e:
            print("[_create_summary] Error:", e, flush=True)
            raise
        
    def write_summary(self):
        print("[write_summary] Starting to write summary...", flush=True)
        try:
            summary_df = self._create_summary()
            filename = os.path.join(self.benchmark_name, "summary.csv")
            print(f"[write_summary] Writing summary to {filename}", flush=True)
            with open(filename, "w") as file:
                for key, value in self.metadata.items():
                    file.write(f"# {key}: {value}\n")
                summary_df.to_csv(file, index=False)
            print("[write_summary] Summary successfully written.", flush=True)
        except Exception as e:
            print("[write_summary] Error:", e, flush=True)
