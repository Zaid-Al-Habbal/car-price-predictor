import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use("dark_background")


def draw_bar_graph(data: pd.DataFrame, feature: str) -> None:

    counts = data[feature].value_counts()

    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45)
    plt.ylabel("count")
    plt.xlabel(feature)
    plt.title(f"{feature} distribution")
    plt.show()


def draw_box_plot(data: pd.DataFrame, feature: str, target: str) -> None:
    plt.figure(figsize=(15, 12))
    plt.boxplot(
        [data[data[feature] == c][target] for c in data[feature].unique()],
        labels=data[feature].unique()
    )
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{target} distribution by {feature}")
    plt.show()