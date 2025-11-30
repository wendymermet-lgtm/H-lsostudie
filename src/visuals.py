import matplotlib.pyplot as plt
import pandas as pd
import src.io_utils as io
from sklearn.linear_model import LinearRegression
import numpy as np

def plot_blood_pressure_histogram(df: pd.DataFrame) -> None:
    """Plots a histogram of blood pressure readings.

    Args:
        df (pd.DataFrame): health data DataFrame.
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the histogram.
    """
    def _draw(ax):
        ax.hist(df["systolic_bp"],color='pink' ,bins=40)
        ax.set_xlabel("Blood pressure in mmHg")
        ax.set_ylabel("Number of participants")
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_title("Blood pressure by participants")

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw(ax)
    plt.tight_layout()

    # attach replot helper so add_to_dashboard can re-draw into a different Axes
    fig.replot = _draw
    plt.close(fig)  # prevent inline display of the original figure
    return fig


def plot_weight_boxplot(df: pd.DataFrame) -> None:
    """Boxplot of weight by gender

    Args:
        df (pd.DataFrame):  health data DataFrame
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the boxplot.
    """

    def _draw(ax):
        df.boxplot(column="weight", by="sex", ax=ax, grid=False)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Weight in kg")
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_title("Boxplot of weight by gender")
        # remove the automatic suptitle that pandas boxplot creates
        try:
            ax.figure.suptitle("")
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig


df = io.df
smoker_df = df[df['smoker'] != 'No']
non_smoker_df = df[df['smoker'] == 'No']
len(smoker_df)
len(non_smoker_df)
#Take df from function arguments
def plot_smoker_bar_chart(df: pd.DataFrame) -> None:
    """Bar chart Smokers, non smokers

    Args:
        df (pd.DataFrame): health data DataFrame
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the bar chart.
    """
    def _draw(ax):
        ax.bar(['Yes', 'No'], [len(smoker_df), len(non_smoker_df)],color=['lightblue', 'lightgreen'])
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Number of participants")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("Bar chart showing number of smokers and non-smokers")

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig
   

def ci_mean_normal_graph(lo, hi, mean_x) -> None:
    """Plots a graph showing the confidence interval for the mean blood pressure using normal approximation.

    Args:
        lo (float): lowest point of the confidence interval
        hi (float): highest point of the confidence interval
        mean_x (float): mean blood pressure
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the confidence interval graph.
    """
    def _draw(ax):
        ax.errorbar([0], [mean_x], yerr=[[mean_x - lo], [hi - mean_x]], fmt="o", capsize=6)
        ax.set_title("95%- CI for the average (normal -approximation)\n for blood pressure")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks([0])
        ax.set_ylabel("Averages in mmHg")
        ax.set_xticklabels(["Blood Pressure (average)"])

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig

def ci_mean_boot_graph(lo, hi, mean_x) -> None:
    """Plots a graph showing the confidence interval for the mean blood pressure using bootstrap method.
    Args:
        lo (float): lowest point of the confidence interval
        hi (float): highest point of the confidence interval
        mean_x (float): mean blood pressure
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the confidence interval graph.
    """
    def _draw(ax):
        ax.errorbar([0], [mean_x], yerr=[[mean_x - lo], [hi - mean_x]], fmt="o", capsize=6)
        ax.set_title("95%- CI for the average (bootstrap)\n for blood pressure")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks([0])
        ax.set_ylabel("Averages in mmHg")
        ax.set_xticklabels(["Blood Pressure (average)"])

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig


def plot_regression_weight_bp(df: pd.DataFrame, predictions) -> None:
    """Plots a scatter plot of blood pressure against weight with a regression line.

    Args:
        df (pd.DataFrame): health data DataFrame containing 'weight' and 'systolic_bp'.
        predictions (np array of float): predicted values from a linear regression model.
    Returns:    
        fig (plt.Figure): Matplotlib Figure object containing the scatter plot and regression line.
    """
    def _draw(ax):
        ax.scatter(df['weight'], df['systolic_bp'], alpha=0.2, label='Blood Pressure per weight', color='purple')
        ax.scatter(df['weight'], predictions, color='hotpink', label='Regression Line')
        ax.set_xlabel('Weight (kg)')
        ax.set_ylabel('Systolic Blood Pressure (mmHg)')
        ax.set_title('Linear Regression: Blood Pressure vs Weight')
        ax.legend()

    fig, ax = plt.subplots()
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig

def plot_regression_age_bp(df: pd.DataFrame, predictions) -> None:
    """Plots a scatter plot of blood pressure against age with a regression line.

    Args:
        df (pd.DataFrame): health data DataFrame containing 'age' and 'systolic_bp'.
        predictions (np array of float): predicted values from a linear regression model.
    Returns: 
        fig (plt.Figure): Matplotlib Figure object containing the scatter plot and regression line.
    """
    def _draw(ax):
        ax.scatter(df['age'], df['systolic_bp'], alpha=0.6, label='Blood Pressure vs Age', color='pink')
        ax.scatter(df['age'], predictions, color='hotpink', alpha=1, label='Regression Line')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Systolic Blood Pressure (mmHg)')
        ax.set_title('Linear Regression: Blood Pressure vs Age')
        ax.legend()

    fig, ax = plt.subplots()
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig

def plot_predictions_bp(df: pd.DataFrame, predictions) -> None:
    """Plots a scatter plot of predicted blood pressure against actual blood pressure.
    Args:
        df (pd.DataFrame): health data DataFrame containing 'systolic_bp'.
        predictions (np array of float): predicted values from a linear regression model.
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the scatter plot.
    """
    def _draw(ax):
        ax.scatter(predictions, df['systolic_bp'], alpha=0.6, label='Predicted Blood Pressure vs Blood Pressure', color='pink')
        # ax.scatter(df['age'], predictions, color='hotpink', alpha=1, label='Regression Line')
        ax.set_xlabel('Predicted Systolic Blood Pressure (mmHg)')
        ax.set_ylabel('Systolic Blood Pressure (mmHg)')
        ax.set_title('Linear Regression: Predicted Blood Pressure vs Blood Pressure')
        ax.legend()

    fig, ax = plt.subplots()
    _draw(ax)
    plt.tight_layout()
    fig.replot = _draw
    plt.close(fig)
    return fig

def add_to_dashboard(figures, cols=2, figsize=(12, 8),title="Dashboard"):
    """Creates a dashboard figure with multiple subplots from a list of figures.
    Args:
        figures (list): List of Matplotlib Figure objects or Axes objects to be added to the dashboard.
        cols (int): Number of columns in the dashboard layout.
        figsize (tuple): Size of the dashboard figure.
        title (str): Title for the dashboard figure.
    Returns:
        None: Displays the dashboard figure with all subplots.
    """
    rows = (len(figures) + cols - 1) // cols
    dashboard_fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, fig in zip(axes, figures):
        # Skip None figures
        if fig is None:
            ax.axis('off')
            continue

        # Support both Figure objects and Axes objects
        orig_ax = None
        # If the figure has a replot helper, use it to draw properly sized content
        if hasattr(fig, 'replot') and callable(getattr(fig, 'replot')):
            ax.clear()
            try:
                fig.replot(ax)
            except Exception:
                ax.axis('off')
            continue

        # Fallback: try to extract original axes and copy artists (best-effort)
        if hasattr(fig, 'axes') and len(fig.axes) > 0:
            orig_ax = fig.axes[0]
        elif hasattr(fig, 'get_children') and isinstance(fig, plt.Axes):
            orig_ax = fig

        if orig_ax is None:
            ax.axis('off')
            continue

        for artist in orig_ax.get_children():
            try:
                ax.add_artist(artist)
            except Exception:
                pass

        try:
            ax.set_xlim(orig_ax.get_xlim())
            ax.set_ylim(orig_ax.get_ylim())
            ax.set_title(orig_ax.get_title())
            ax.set_xlabel(orig_ax.get_xlabel())
            ax.set_ylabel(orig_ax.get_ylabel())
        except Exception:
            pass

    # Hide unused axes
    for ax in axes[len(figures):]:
        ax.axis("off")
    
    # set title on the dashboard figure (not on the last source fig)
    dashboard_fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    return None