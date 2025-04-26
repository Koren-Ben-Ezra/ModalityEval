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
        self.logger = Log().logger

        self.benchmark_name = os.path.join(
            BENCHMARKS_DIR, 
            self.metadata["Model"], 
            self.metadata["Dataset"]
        )
        
        self._make_benchmark_dir()
        self._make_summary_files()
    
    def register_job(self, text_f: AbstractTextFilter=None, img_f: AbstractImageFilter=None, inner_dir: str=""):
        if img_f is None and text_f is None:
            raise ValueError("Both image and text filters cannot be None.")
        
        JobHandler().register_job(partial(self._execute_test, text_f, img_f, inner_dir))

    def start_workers(self):
        JobHandler().start_workers()
        
    def _execute_test(self, text_f: AbstractTextFilter=None, img_f: AbstractImageFilter=None, inner_dir: str=""):
        if text_f is None and img_f is None:
            raise ValueError("Both text and image filters cannot be None.")

        name_tf = text_f.filter_name if text_f else ""
        name_if = img_f.filter_name if img_f else ""
        
        if name_tf and name_if:
            self.logger.info(f"Executing test with filters: {name_tf} and {name_if}")
        elif name_tf:
            self.logger.info(f"Executing test with filter: {name_tf}")
        elif name_if:
            self.logger.info(f"Executing test with filter: {name_if}")
            
        category = Category(text_f, img_f, self.benchmark_name, inner_dir)

        idx = 0
        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category, idx)
            idx += 1
            
        category.save_predictions()
        self.append_res_to_summary(category)
    
    def _track_result(self, question: str, answer:  str, final_pred: str, full_pred: str, title: str):
        predictions_filename = os.path.join(self.benchmark_name, "track.txt")
        if not os.path.exists(predictions_filename):
            open(predictions_filename, "w").close()

        with open(predictions_filename, "a") as f:
            f.write("-------------------------------------------------------------\n")
            f.write(f"[{title}]\n\n")
            f.write(f"question: {question}\n")
            f.write(f"answer: {answer}\n")
            f.write(f"final pred: {final_pred}\n")
            f.write(f"full pred: '{full_pred}'\n")

        
    def _execute_single_prompt(self, sample, category: Category, idx: int):
        final_pred_txt, full_pred_txt = None, None
        final_pred_img, full_pred_img = None, None
        if category.text_f is not None:
            txtf_title = f"filter: {category.text_f.filter_name}, question: {idx}"
            try:
                filtered_text = category.text_f.apply_filter(sample["question"], sample["answer"])
            except Exception as e:
                self.logger.error(f"Error applying text filter: {e}")
                raise e
            try:
                final_pred_txt, full_pred_txt = self.multimodal_wrapper.generate_ans_from_text(filtered_text)
            except Exception as e:
                self.logger.error(f"Error generating answer from text: {e}")
                raise e

        if category.img_f is not None:
            imgf_title = f"filter: {category.img_f.filter_name}, question: {idx}"
            try:
                filtered_image = category.img_f.apply_filter(sample["question_image"], sample["answer"])
            except Exception as e:
                self.logger.error(f"Error applying image filter: {e}")
                raise e
            try:
                final_pred_img, full_pred_img = self.multimodal_wrapper.generate_ans_from_image(filtered_image)
            except Exception as e:
                self.logger.error(f"Error generating answer from image: {e}")
                raise e
            
        self._update_category_stats(sample, category, final_pred_txt, final_pred_img, full_pred_txt, full_pred_img, txtf_title, imgf_title)
    
    def _update_category_stats(self, sample, category: Category, final_pred_txt: str, final_pred_img: str, full_pred_txt: str, full_pred_img: str, txtf_title: str, imgf_title: str):
        answer = BenchmarkManager.clean_str_number(sample["answer"])

        category.text_stats.total += 1
        category.img_stats.total += 1

        if final_pred_txt:
            if final_pred_txt == answer:
                category.text_stats.success += 1
            else:
                self._track_result(sample["question"], sample["answer"], final_pred_txt, full_pred_txt, txtf_title)
        
        if final_pred_img:
            if final_pred_img == answer:
                category.img_stats.success += 1
            else:
                self._track_result(sample["question"], sample["answer"], final_pred_img, full_pred_img, imgf_title)

        # Append predictions if predictions_df exists
        if category.predictions_df is not None:
            new_row = pd.DataFrame([{
                "answer": answer,
            }])
            if final_pred_txt is not None:
                new_row["text_prediction"] = final_pred_txt

            if final_pred_img is not None:
                new_row["img_prediction"] = final_pred_img            

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
