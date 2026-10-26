import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import df

def plot_marks_distribution(df):
    """
        This function displays the distribution of student marks
    """
    plt.figure(figsize = (8,5))

    sns.histplot(data = df, x = "Marks", bins = 10, kde = True)

    plt.title("Distribution of student marks")
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()

def plot_study_hours_vs_marks(df):
    """
        Display relationship between study hours and marks
    """
    plt.figure(figsize = (8,5))

    sns.scatterplot(data = df, x = "StudyHours", y = "Marks")

    plt.title("Study Hours vs Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """
        Display correlations between numerical columns
    """
    plt.figure(figsize = (8,6))

    correlation = df.corr(numeric_only = True)

    sns.heatmap(correlation, annot = True, cmap = "coolwarm", fmt = ".2f", vmin = -1, vmax = 1)

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    """
        Display box plot for numerical columns
    """
    columns = ["StudyHours", "Attendence","Assignments","Marks"]

    for column in columns:
        plt.figure(figsize = (7,4))
        sns.boxplot(data = df, x = column)
        plt.title(f"Box plot of {column}")
        plt.tight_layout()
        plt.show()

plot_marks_distribution(df)
plot_study_hours_vs_marks(df)
plot_correlation_heatmap(df)
plot_boxplots(df)