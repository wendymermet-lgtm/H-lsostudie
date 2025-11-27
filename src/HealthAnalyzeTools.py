import src.io_utils as io
import src.metrics as mt
import numpy as np
import src.visuals as vis
from src.io_utils import df
from math import sqrt
from scipy import stats
import statsmodels.stats.power as power_analysis
from sklearn.linear_model import LinearRegression

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
        return vis.plot_blood_pressure_histogram(df=self.df)

    def plot_weight_boxplot(self) -> None:
        return vis.plot_weight_boxplot(df=self.df)

    def plot_smoker_bar_chart(self) -> None:
        return vis.plot_smoker_bar_chart(df=self.df)

class HealthStatistics:
    def __init__(self):
        df = io.load_data("Data\health_study_dataset.csv")
        self.df = df

    def perc_sick_participants(self) -> float:
        sick_df = df[df['disease'] != 0]
        return len(sick_df)/len(df)
    
    
    def simulate_disease(self, n: int = 1000) -> tuple[float, float]:
        np.random.seed(42)
        mean_sick_df = self.df["disease"].mean()
        diseases = np.random.choice([0, 1], size=n, p=[1 - mean_sick_df, mean_sick_df])
        mean_diseases = diseases.mean()
        return mean_diseases, mean_sick_df

    np.random.seed(42)
    x = np.random.choice(df["systolic_bp"], size = 5000, replace = True)

    def ci_mean_normal (self, x, confidence= 0.95):
        x = np.asarray(x, dtype = float)
        mean_x = float(np.mean(x))
        s = float(np.std(x,ddof = 1))
        n= len(x)
        z_critical = 1.96
        half_width = z_critical * s  / sqrt (n)
        lo, hi = mean_x - half_width, mean_x + half_width
        return lo, hi, mean_x, s, n
    
    def ci_mean_normal_graph(self, lo, hi, mean_x) -> None:
        return vis.ci_mean_normal_graph(lo, hi, mean_x)

    def ci_mean_bootstrap(self, x,B= 5_000,confidence=0.95):
        x = np.asarray(x, dtype = float)
        n= len(x)
        boot_means = np.empty(B)
        for b in range (B):
            boot_sample=np.random.choice(x,size=n, replace= True)
            boot_means[b] = np.mean(boot_sample)
        alpha = (1 - confidence) / 2
        lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
        mean_x = np.mean(x)
        return float(lo), float(hi), float(mean_x)
    
    def ci_mean_boot_graph(self, lo, hi, mean_x) -> None:
        return vis.ci_mean_boot_graph(lo, hi, mean_x)

     
    def hypothesis_test_smoker_bp(self):
        A = df.loc[df.smoker == "Yes", "systolic_bp"].values
        B = df.loc[df.smoker == "No", "systolic_bp"].values
        mu_A = np.mean(A)
        mu_B = np.mean(B)
        true_diff = mu_A - mu_B  
        np.random.seed(42)
        n_boot = 5_000
        boot_diff = np.empty(n_boot)
        for b in range (n_boot):
            boot_A = np.random.choice(A, size = len(A), replace = True)
            boot_B = np.random.choice(B, size = len(B), replace = True)
            boot_diff[b] = np.mean(boot_A) - np.mean(boot_B)

        p_boot = np.mean(np.abs(boot_diff) >= np.abs(true_diff))
        cilow, cihigh = np.percentile(boot_diff, [2.5, 97.5])
        return p_boot, cilow, cihigh, true_diff
    

    def cohen_d_smoker_bp(self):
        A = df.loc[df.smoker == "Yes", "systolic_bp"].values
        B = df.loc[df.smoker == "No", "systolic_bp"].values
        mu_A = np.mean(A)
        mu_B = np.mean(B)
        sd_pooled = np.sqrt(((len(A) - 1) * np.var(A, ddof=1) + (len(B) - 1) * np.var(B, ddof=1)) / (len(A) + len(B) - 2))
        d = (mu_A - mu_B) / sd_pooled
        n_per_group = min(len(A), len(B))
        return d, n_per_group
    
    def plot_regression_weight_bp(self, predictions) -> None:
        return vis.plot_regression_weight_bp(df=self.df, predictions=predictions)
    
    def plot_regression_age_bp(self, predictions) -> None:
        return vis.plot_regression_age_bp(df=self.df, predictions=predictions)
    
    def linear_regression_age_weight_bp(self):
        x_2d = df[['age', 'weight']].values
        y = df['systolic_bp'].values

        model = LinearRegression()
        model.fit(x_2d, y)
        intercept= model.intercept_
        coef_age= model.coef_[0]
        coef_weight= model.coef_[1]
        r_squared = model.score(x_2d, y)
        predictions = model.predict(x_2d)
        return intercept, coef_age, coef_weight, r_squared, predictions
        
    def corr_age_weight_bp(self):
        corr_age_bp = np.corrcoef(df['age'], df['systolic_bp'])[0, 1]
        corr_weight_bp = np.corrcoef(df['weight'], df['systolic_bp'])[0, 1]
        return corr_age_bp, corr_weight_bp







 


