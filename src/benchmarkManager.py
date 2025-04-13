import os
import pandas as pd
from tqdm import tqdm

from src.filters import AbstractImageFilter, AbstractTextFilter
from src.category import Category
from src.datasetWrapper import AbstractDatasetWrapper
from src.multimodalWrappers import MultimodalWrapper
from src.log import Log

BENCHMARKS_DIR = "benchmarks"
EPSILON = 1e-3

class BenchmarkManager:
    def __init__(self, datasetWrapper: AbstractDatasetWrapper, multimodal_wrapper: MultimodalWrapper, metadata: dict):
        self._datasetWrapper = datasetWrapper
        self.multimodal_wrapper = multimodal_wrapper
        self.metadata = metadata
        self.save_predictions = metadata.get("Save Predictions", False)
        self.logger = Log().logger
        self._categories: list[Category] = []

        self.benchmark_name = os.path.join(
            BENCHMARKS_DIR, 
            self.metadata["Model"], 
            self.metadata["Dataset"]
        )
        
        self._make_benchmark_dir()

        
    def execute_test(self, text_f: AbstractTextFilter=None, img_f: AbstractImageFilter=None):
        if text_f is None and img_f is None:
            raise ValueError("Both text and image filters cannot be None.")

        name1 = text_f.filter_name if text_f is not None else ""
        name2 = img_f.filter_name if img_f is not None else ""
        if name1 and name2:
            self.logger.info(f"Executing test with filters: {name1} and {name2}")
        elif name1:
            self.logger.info(f"Executing test with filter: {name1}")
        elif name2:
            self.logger.info(f"Executing test with filter: {name2}")
            
        category = Category(text_f, img_f, self.benchmark_name, self.save_predictions)
        self._categories.append(category)

        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category)
            
        category.save_predictions()
            
    def _execute_single_prompt(self, sample, category: Category):
        
        pred_from_text = None
        pred_from_image = None
        if category.text_f is not None:
            filtered_text = category.text_f.apply_filter(sample["question"])
            pred_from_text = self.multimodal_wrapper.generate_ans_from_text(filtered_text)
            
        if category.img_f is not None:
            filtered_image = category.img_f.apply_filter(sample["question_image"])
            pred_from_image = self.multimodal_wrapper.generate_ans_from_image(filtered_image)
        
        self._update_category_stats(sample, category, pred_from_text, pred_from_image)
    
    def _update_category_stats(self, sample, category: Category, pred_from_text: str, pred_from_img: str):

        answer = sample["answer"]
        
        if pred_from_text is not None:
            category.text_stats.total += 1
            if pred_from_text == answer:
                category.text_stats.success += 1
        
        if pred_from_img is not None:        
            category.img_stats.total += 1
            if pred_from_img == answer:
                category.img_stats.success += 1

        # Append predictions if predictions_df exists
        if category.predictions_df is not None:
            new_row = pd.DataFrame([{
                "answer": answer,
            }])
            if pred_from_text is not None:
                new_row["text_prediction"] = pred_from_text

            if pred_from_img is not None:
                new_row["img_prediction"] = pred_from_img            

            category.predictions_df = pd.concat([category.predictions_df, new_row],ignore_index=True)
    
    def _make_benchmark_dir(self):
        if not os.path.exists(BENCHMARKS_DIR):
            os.makedirs(BENCHMARKS_DIR)
        if not os.path.exists(self.benchmark_name):
            os.makedirs(self.benchmark_name)
    
    def _create_summary(self):
        columns = ["Text Filter", "Text Correct", "Image Filter", "Image Correct", "Total Samples"]
        
        summary_df = pd.DataFrame(columns=columns)
        for category in self._categories:
            new_row = pd.DataFrame([{}])
            
            new_row["Text Filter"] = ""
            new_row["Text Correct"] = ""
            new_row["Image Filter"] = ""
            new_row["Image Correct"] = ""
            
            if category.text_f is not None:
                new_row["Text Filter"] = category.text_f.filter_name
                new_row["Text Correct"] = category.text_stats.success
            if category.img_f is not None:
                new_row["Image Filter"] = category.img_f.filter_name
                new_row["Image Correct"] = category.img_stats.success
            
            new_row["Total Samples"] = category.text_stats.total
            summary_df = pd.concat([summary_df, new_row], ignore_index=True)
            
        return summary_df
        
    def write_summary(self):
        # print("[write_summary] Starting to write summary...", flush=True)
        try:
            self.logger.info("Writing summary...")
            summary_df = self._create_summary()
            filename = self.metadata.get("test name", "summary").replace(" ", "_") + ".csv"
            filename = os.path.join(self.benchmark_name, filename)
            with open(filename, "w") as file:
                for key, value in self.metadata.items():
                    file.write(f"# {key}: {value}\n")
                summary_df.to_csv(file, index=False)
            self.logger.info(f"Summary successfully written to {filename}.")
        except Exception as e:
            self.logger.error("Error writing summary:", e)
