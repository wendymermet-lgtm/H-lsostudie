import src.io_utils as io
import src.metrics as mt
import numpy as np
import src.visuals as vis
from src.io_utils import df

class HealthAnalyzer:
    def __init__(self):
        df = io.load_data("Data\health_study_dataset.csv")
        self.df = df
    

    def main_summary(self):
        age = io.df["age"].to_numpy()
        print("In column \"age\":")
        mt.statistics(age)

        height = io.df["height"].to_numpy()
        print("In column \"height\":")
        mt.statistics(height)

        weight = io.df["weight"].to_numpy()
        print("In column \"weight\":")
        mt.statistics(weight)

        systolic_bp = io.df["systolic_bp"].to_numpy()
        print("In column \"systolic_bp\":")
        mt.statistics(systolic_bp)

        cholesterol = io.df["cholesterol"].to_numpy()
        print("In column \"cholesterol\":")
        mt.statistics(cholesterol)
        
    def plot_blood_pressure_histogram(self) -> None:
        vis.plot_blood_pressure_histogram(df=df)

    def plot_weight_boxplot(self) -> None:
        vis.plot_weight_boxplot(df=df)

    def plot_smoker_bar_chart(self) -> None:
        vis.plot_smoker_bar_chart(df=df)

   

