import os
import pandas as pd

from src.filters import AbstractTextFilter, AbstractImageFilter
from src.log import Log

class Category:
    class Statistics:
        def __init__(self):
            self.success = 0
            self.total = 0

    def __init__(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter, benchmark_name: str, inner_dir: str):
        self.text_f = text_f
        self.img_f = img_f
        
        if inner_dir:
            temp_path = os.path.join(benchmark_name, inner_dir)
        else:
            temp_path = benchmark_name
            
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        if text_f is not None and img_f is not None:
            self.category_name = os.path.join(temp_path, f"{text_f.filter_name}_{img_f.filter_name}")
            self.predictions_df = pd.DataFrame(columns=["answer", "text_prediction", "img_prediction"])

        elif text_f is not None:
            self.category_name = os.path.join(temp_path, text_f.filter_name)
            self.predictions_df = pd.DataFrame(columns=["answer", "text_prediction"])

        else:
            self.category_name = os.path.join(temp_path, img_f.filter_name)
            self.predictions_df = pd.DataFrame(columns=["answer", "img_prediction"])
    
        self.text_stats = Category.Statistics()
        self.img_stats = Category.Statistics()
        
    def save_predictions(self):
        filename = self.category_name + ".csv"
        self.predictions_df.to_csv(filename, index=True)
        Log().logger.info(f"Saved predictions to {filename}")
        