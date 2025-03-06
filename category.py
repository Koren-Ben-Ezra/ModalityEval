import pandas as pd

from filters import AbstractTextFilter, AbstractImageFilter


class Category:
    class Statistics:
        def __init__(self):
            self.success = 0
            self.total = 0

    def __init__(self, text_f: AbstractTextFilter, img_f: AbstractImageFilter, benchmark_name: str, save_predictions: bool):
        self.text_f = text_f
        self.img_f = img_f
        self.category_name = benchmark_name + "\\" + f"{text_f.filter_name}_{img_f.filter_name}"
        
        self.text_stats = Category.Statistics()
        self.img_stats = Category.Statistics()
        self.predictions_df = None
        if save_predictions:
            self.predictions_df = pd.DataFrame(columns=["answer", "text_prediction", "img_prediction"])

    def eval_accuracy(self):
        text_acc = self.text_stats.success / self.text_stats.total
        img_acc = self.img_stats.success / self.img_stats.total
        return text_acc, img_acc
    
    def save_predictions(self):
        if self.predictions_df is not None:
            filename = self.category_name + ".csv"
            self.predictions_df.to_csv(filename, index=True)