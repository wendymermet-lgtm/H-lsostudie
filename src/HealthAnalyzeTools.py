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
        """Initializes the HealthAnalyzer class and loads the health study dataset.
        This method loads the dataset from a CSV file and stores it in the instance variable `df`.
        Returns:
            None
        """
        df = io.load_data("Data\health_study_dataset.csv")
        self.df = df
    

    def main_summary(self):
        """Prints summary statistics for the health study dataset.
        This method calculates and prints the mean, standard deviation, minimum, maximum, and count for the columns:
        'age', 'height', 'weight', 'systolic_bp', and 'cholesterol'.
        Returns:
            None
        """
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
        """Plots a histogram of the systolic blood pressure from the health study dataset.
        This method uses the `plot_blood_pressure_histogram` function from the `visuals` module to create a histogram.
        Returns:
            None: Displays the histogram plot.
        """
        return vis.plot_blood_pressure_histogram(df=self.df)

    def plot_weight_boxplot(self) -> None:
        """Plots a boxplot of the weight from the health study dataset.
        This method uses the `plot_weight_boxplot` function from the `visuals` module to create a boxplot.
        Returns:
            None: Displays the boxplot.
        """
        return vis.plot_weight_boxplot(df=self.df)

    def plot_smoker_bar_chart(self) -> None:
        """Plots a bar chart showing the number of smokers and non-smokers in the health study dataset.
        This method uses the `plot_smoker_bar_chart` function from the `visuals` module to create the bar chart.
        Returns:
            None: Displays the bar chart.
        """
        return vis.plot_smoker_bar_chart(df=self.df)

class HealthStatistics:
    def __init__(self):
        """Initializes the HealthStatistics class and loads the health study dataset.
        This method loads the dataset from a CSV file and stores it in the instance variable `df`.
        Returns:
            None
        """
        df = io.load_data("Data\health_study_dataset.csv")
        self.df = df

    def perc_sick_participants(self) -> float:
        """Calculates the percentage of participants with a disease in the health study dataset.
        This method filters the dataset to find participants with a disease (where 'disease' is
        not equal to 0) and calculates the percentage of such participants relative to the total number of participants.
        Returns:
            float: Percentage of participants with a disease.
        """
        sick_df = df[df['disease'] != 0]
        return len(sick_df)/len(df)
    
    
    def simulate_disease(self, n: int = 1000) -> tuple[float, float]:
        """Simulates the occurrence of a disease in a population of size n.
        This method generates a random sample of disease occurrences based on the mean disease rate in the dataset.
        Args:
            n (int, optional): Size of the population to simulate. Defaults to 1000
        Returns:
            tuple: A tuple containing the mean of the simulated diseases and the mean of the actual disease
            occurrences in the dataset.
        """
        np.random.seed(42)
        mean_sick_df = self.df["disease"].mean()
        diseases = np.random.choice([0, 1], size=n, p=[1 - mean_sick_df, mean_sick_df])
        mean_diseases = diseases.mean()
        return mean_diseases, mean_sick_df

    np.random.seed(42)
    x = np.random.choice(df["systolic_bp"], size = 5000, replace = True)

    def ci_mean_normal (self, x, confidence= 0.95):
        """Calculates the confidence interval for the mean of a sample using the normal approximation.
        Args:
            x (array-like): Sample data for which to calculate the confidence interval.
            confidence (float, optional): Confidence level for the interval. Defaults to 0.95.
        Returns:
            tuple: A tuple containing the lower bound, upper bound, mean, standard deviation, and sample size.
        """
        x = np.asarray(x, dtype = float)
        mean_x = float(np.mean(x))
        s = float(np.std(x,ddof = 1))
        n= len(x)
        z_critical = 1.96
        half_width = z_critical * s  / sqrt (n)
        lo, hi = mean_x - half_width, mean_x + half_width
        return lo, hi, mean_x, s, n
    
    def ci_mean_normal_graph(self, lo, hi, mean_x) -> None:
        """Creates a graph to visualize the confidence interval for the mean.
        Args:
            lo (float): Lower bound of the confidence interval.
            hi (float): Upper bound of the confidence interval.
            mean_x (float): Mean of the sample data.
        Returns:
            None: Displays the graph showing the confidence interval.
        """
        return vis.ci_mean_normal_graph(lo, hi, mean_x)

    def ci_mean_bootstrap(self, x,B= 5_000,confidence=0.95):
        """Calculates the confidence interval for the mean of a sample using bootstrap resampling.
        Args:
            x (array-like): Sample data for which to calculate the confidence interval.
            B (int, optional): Number of bootstrap samples to generate. Defaults to 5,000.
            confidence (float, optional): Confidence level for the interval. Defaults to 0.95.
        Returns:
            tuple: A tuple containing the lower bound, upper bound, and mean of the bootstrap samples.
        """
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
        """Creates a graph to visualize the confidence interval for the mean using bootstrap resampling.
        Args:
            lo (float): Lower bound of the confidence interval.
            hi (float): Upper bound of the confidence interval.
            mean_x (float): Mean of the sample data.
        Returns:
            None: Displays the graph showing the confidence interval.
        """
        return vis.ci_mean_boot_graph(lo, hi, mean_x)

     
    def hypothesis_test_smoker_bp(self):
        """Performs a hypothesis test to compare the mean systolic blood pressure between smokers and non-smokers.
        This method uses bootstrap resampling to estimate the p-value and confidence interval for the difference in means.
        Returns:
            tuple: A tuple containing the p-value, lower bound, upper bound of the confidence interval, and the true difference in means.
        """

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
        """Calculates Cohen's d for the difference in mean systolic blood pressure between smokers and non-smokers.
        This method computes the means and pooled standard deviation for the two groups and returns the effect size.
        Returns:
            tuple: A tuple containing Cohen's d and the number of participants in each group.
        """
        A = df.loc[df.smoker == "Yes", "systolic_bp"].values
        B = df.loc[df.smoker == "No", "systolic_bp"].values
        mu_A = np.mean(A)
        mu_B = np.mean(B)
        sd_pooled = np.sqrt(((len(A) - 1) * np.var(A, ddof=1) + (len(B) - 1) * np.var(B, ddof=1)) / (len(A) + len(B) - 2))
        d = (mu_A - mu_B) / sd_pooled
        n_per_group = min(len(A), len(B))
        return d, n_per_group
    
    def plot_regression_weight_bp(self, predictions) -> None:
        """Plots a regression line of weight against systolic blood pressure.
        Args:
            predictions (np array of float): Predicted values from a linear regression model.
        Returns:
            fig (plt.Figure): Matplotlib Figure object containing the scatter plot and regression line.
        """
        return vis.plot_regression_weight_bp(df=self.df, predictions=predictions)
    
    def plot_regression_age_bp(self, predictions) -> None:
        """Plots a regression line of age against systolic blood pressure.
        Args:
            predictions (np array of float): Predicted values from a linear regression model.
        Returns:
            fig (plt.Figure): Matplotlib Figure object containing the scatter plot and regression line.
        """
        return vis.plot_regression_age_bp(df=self.df, predictions=predictions)
    
    def plot_predictions_bp(self, predictions) -> None:
        """Plots a scatter plot of predicted blood pressure against actual blood pressure.
        Args:
            predictions (np array of float): Predicted values from a linear regression model.
        Returns:
            fig (plt.Figure): Matplotlib Figure object containing the scatter plot.
        """
        return vis.plot_predictions_bp(df=self.df, predictions=predictions)
    
    def linear_regression_age_weight_bp(self):
        """Performs linear regression to predict systolic blood pressure based on age and weight.
        This method fits a linear regression model using age and weight as predictors and returns the model coefficients,
        intercept, R-squared value, and predictions.
        Returns:
            tuple: A tuple containing the intercept, coefficient for age, coefficient for weight, R-squared value, and predictions.
        """
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
        """Calculates the correlation coefficients between age, weight, and systolic blood pressure.
        This method computes the Pearson correlation coefficients between age and systolic blood pressure,
        and between weight and systolic blood pressure.
        Returns:
            tuple: A tuple containing the correlation coefficient between age and systolic blood pressure,
            and the correlation coefficient between weight and systolic blood pressure.
        """
        corr_age_bp = np.corrcoef(df['age'], df['systolic_bp'])[0, 1]
        corr_weight_bp = np.corrcoef(df['weight'], df['systolic_bp'])[0, 1]
        return corr_age_bp, corr_weight_bp







 


