import os
import pandas as pd
from tqdm import tqdm
import json
from decimal import Decimal, InvalidOperation
from functools import partial

from src.filters import AbstractImageFilter, AbstractTextFilter
from src.category import Category
from src.datasetWrapper import AbstractDatasetWrapper
from src.multimodalWrappers import MultimodalWrapper
from src.log import Log
from src.job_handler import JobHandler

BENCHMARKS_DIR = "benchmarks"
EPSILON = 1e-3

class BenchmarkManager:
    def __init__(self, datasetWrapper: AbstractDatasetWrapper, multimodal_wrapper: MultimodalWrapper, metadata: dict):
        self._datasetWrapper = datasetWrapper
        self.multimodal_wrapper = multimodal_wrapper
        self.metadata = metadata
        self.save_predictions = metadata.get("Save Predictions", False)
        self.logger = Log().logger

        self.benchmark_name = os.path.join(
            BENCHMARKS_DIR, 
            self.metadata["Model"], 
            self.metadata["Dataset"]
        )
        
        self._make_benchmark_dir()
        self._make_summary_files()
    
    def add_job(self, img_f: AbstractImageFilter=None, text_f: AbstractTextFilter=None):
        if img_f is None and text_f is None:
            raise ValueError("Both image and text filters cannot be None.")
        
        JobHandler().register_job(partial(self._execute_test, text_f, img_f))

    def _execute_test(self, text_f: AbstractTextFilter=None, img_f: AbstractImageFilter=None):
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

        cnt = 0
        for sample in tqdm(self._datasetWrapper.dataset):
            track_result = cnt < 5 or cnt % 50 == 0
            self._execute_single_prompt(sample, category, track_result)
            
        category.save_predictions()
        self.append_res_to_summary(category)
    
    def _track_result(self, question: str, answer: str, pred_from_text: str, pred_from_image: str):
        new_row = pd.DataFrame([{
            "question": question,
            "answer": answer
        }])
        if pred_from_text is not None:
            new_row["text_prediction"] = pred_from_text
        if pred_from_image is not None:
            new_row["img_prediction"] = pred_from_image
        
        predictions_filename = os.path.join(self.benchmark_name, "debug.csv")
        new_row.to_csv(predictions_filename, mode='a', index=False, header=not os.path.exists(predictions_filename))
        
    def _execute_single_prompt(self, sample, category: Category, track_result: bool = False):
        pred_from_text = None
        pred_from_image = None
        if category.text_f is not None:
            try:
                filtered_text = category.text_f.apply_filter(sample["question"])
            except Exception as e:
                self.logger.error(f"Error applying text filter: {e}")
                raise e
            
            try:
                pred_from_text = self.multimodal_wrapper.generate_ans_from_text(filtered_text)
            except Exception as e:
                self.logger.error(f"Error generating answer from text: {e}")
                raise e
            
        if category.img_f is not None:
            try:
                filtered_image = category.img_f.apply_filter(sample["question_image"])
            except Exception as e:
                self.logger.error(f"Error applying image filter: {e}")
                raise e
            
            try:
                pred_from_image = self.multimodal_wrapper.generate_ans_from_image(filtered_image)
            except Exception as e:
                self.logger.error(f"Error generating answer from image: {e}")
                raise e
            
        self._update_category_stats(sample, category, pred_from_text, pred_from_image)
    
        if track_result:
            self._track_result(sample["question"], sample["answer"], pred_from_text, pred_from_image)
    
    def _update_category_stats(self, sample, category: Category, pred_from_text: str, pred_from_img: str):
        answer = self.clean_str_number(sample["answer"])
        
        if pred_from_text is not None:
            category.text_stats.total += 1
            pred_from_text = self.clean_str_number(pred_from_text)    
            if pred_from_text == answer:
                category.text_stats.success += 1
        
        if pred_from_img is not None:        
            category.img_stats.total += 1
            pred_from_text = self.clean_str_number(pred_from_img)
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
    
    def append_res_to_summary(self, category: Category):
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

        with open(self.summary_filename, "a") as f:
            new_row.to_csv(f, index=False, header=False)

    def _make_summary_files(self):
        columns = ["Text Filter", "Text Correct", "Image Filter", "Image Correct", "Total Samples"]
        summary_df = pd.DataFrame(columns=columns)
        self.summary_filename = os.path.join(self.benchmark_name, self.metadata.get("test name", "summary").replace(" ", "_") + ".csv")
        summary_df.to_csv(self.summary_filename, index=False)
        
        metadata_filename = self.summary_filename.replace(".csv", "_metadata.json")
        with open(metadata_filename, "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    @staticmethod
    def clean_str_number(s: str) -> str:
        if not s:
            return s
        
        try:
            d = Decimal(s)
        except InvalidOperation:
            return s
        
        d_normalized = d.normalize()
        s_formatted = format(d_normalized, 'f')
        
        if '.' in s_formatted:
            s_formatted = s_formatted.rstrip('0').rstrip('.')
        
        return s_formatted
