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
TRACK_MISTAKES=True
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
            
        category = Category(text_f, img_f, self.benchmark_name, self.save_predictions, inner_dir)

        idx = 0
        for sample in tqdm(self._datasetWrapper.dataset):
            self._execute_single_prompt(sample, category, idx, track_mistakes=TRACK_MISTAKES)
            idx += 1
            
        category.save_predictions()
        self.append_res_to_summary(category)
    
    def _track_result(self, question: str, answer: str, extracted_pred: str, response: str, title: str, is_correct: bool):
        predictions_filename = os.path.join(self.benchmark_name, "track.txt")
        if not os.path.exists(predictions_filename):
            open(predictions_filename, "w").close()

        with open(predictions_filename, "a") as f:
            f.write("-------------------------------------------------------------\n")
            f.write(f"[{title}] ({'CORRECT' if is_correct else 'WRONG'})\n\n")
            f.write(f"Q: {question}\n")
            f.write(f"Ans: {answer}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Pred: '{extracted_pred}'\n")

        
    def _execute_single_prompt(self, sample, category: Category, idx: int, track_mistakes: bool = False):
        pred_from_txt = None
        pred_from_img = None
        txt_response = None
        img_response = None
        if category.text_f is not None:
            try:
                filtered_text = category.text_f.apply_filter(sample["question"], sample["answer"])
            except Exception as e:
                self.logger.error(f"Error applying text filter: {e}")
                raise e
            try:
                txt_response = self.multimodal_wrapper.generate_ans_from_text(filtered_text)
            except Exception as e:
                self.logger.error(f"Error generating answer from text: {e}")
                raise e
            try:
                pred_from_txt = self.multimodal_wrapper.extract_answer(txt_response) # function may differ in different models
            except Exception as e:
                self.logger.error(f"Error extracting pred_from_text: {e}")
                raise e
        
            txtf_title = f"filter: {category.text_f.filter_name}, question: {idx}"

        if category.img_f is not None:
            try:
                filtered_image = category.img_f.apply_filter(sample["question_image"], sample["answer"])
            except Exception as e:
                self.logger.error(f"Error applying image filter: {e}")
                raise e
            try:
                img_response = self.multimodal_wrapper.generate_ans_from_image(filtered_image)
            except Exception as e:
                self.logger.error(f"Error generating answer from image: {e}")
                raise e
            try:
                pred_from_img = self.multimodal_wrapper.extract_answer(img_response) # function may differ in different models
            except Exception as e:
                self.logger.error(f"Error extracting pred_from_img: {e}")
                raise e
            
            imgf_title = f"filter: {category.img_f.filter_name}, question: {idx}"
            
        self._update_category_stats(sample, category, pred_from_txt, pred_from_img,txt_response, img_response, txtf_title, imgf_title, track_mistakes)
    
    def _update_category_stats(self, sample, category: Category, pred_from_txt: str, pred_from_img: str, txt_response: str, img_response: str, txtf_title: str, imgf_title: str, track_mistakes: bool):
        answer = BenchmarkManager.clean_str_number(sample["answer"])
        
        category.text_stats.total += 1
        category.img_stats.total += 1

        if pred_from_txt is not None:
            is_correct = (pred_from_txt == answer)
            if is_correct:
                category.text_stats.success += 1
            elif track_mistakes:
                self._track_result(sample["question"], sample["answer"], pred_from_txt, txt_response, txtf_title, is_correct)
        
        if pred_from_img is not None:
            is_correct = (pred_from_img == answer)
            if is_correct:
                category.img_stats.success += 1
            elif track_mistakes:
                self._track_result(sample["question"], sample["answer"], pred_from_img, img_response, imgf_title, is_correct)

        # Append predictions if predictions_df exists
        if category.predictions_df is not None:
            new_row = pd.DataFrame([{
                "answer": answer,
            }])
            if pred_from_txt is not None:
                new_row["text_prediction"] = pred_from_txt

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
