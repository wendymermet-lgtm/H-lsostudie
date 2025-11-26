import matplotlib.pyplot as plt
import pandas as pd
import src.io_utils as io

def plot_blood_pressure_histogram(df: pd.DataFrame) -> None:
    """Plots a histogram of systolic blood pressure."""
    fig, ax = plt.subplots(figsize = (10,6))
    ax.hist(df["systolic_bp"],bins =40)
    ax.set_xlabel("Blood pressure in mmHg")
    ax.set_ylabel("Number of participants")
    ax.grid(True, axis = "y")
    ax.set_title("Blood pressure by participants")
    plt.tight_layout()


def plot_weight_boxplot(df: pd.DataFrame) -> None:
    """Plots a boxplot of weight by gender."""
    fig, ax = plt.subplots(figsize = (10,6))
    df.boxplot(column = "weight", by = "sex",ax=ax)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Weight in kg")
    ax.grid(True, alpha=0.2 ,axis = "y")
    ax.set_title("Boxplot of weight by gender")
    plt.suptitle("")
    plt.tight_layout()


df = io.df
smoker_df = df[df['smoker'] != 'No']
non_smoker_df = df[df['smoker'] == 'No']
len(smoker_df)
len(non_smoker_df)

def plot_smoker_bar_chart(df: pd.DataFrame) -> None:
    """Plots a bar chart showing number of smokers and non-smokers."""
    fig, ax = plt.subplots(figsize = (10,6))
    ax.bar(['Yes', 'No'], [len(smoker_df), len(non_smoker_df)])
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Number of participants")
    ax.grid(True, alpha = 0.3, axis = "y")
    ax.set_title("Bar chart showing number of smokers and non-smokers")
    plt.tight_layout()
   

def dashboard(g1,g2,g3) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 5), sharex=False, sharey=False)
    axes[0,0] = g1
    axes[0,1] = g2
    axes[1,0] = g3
    axes[1,1].axis('off')  # Turn off unused subplot axis
    fig.suptitle("Health Study Dashboard", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
