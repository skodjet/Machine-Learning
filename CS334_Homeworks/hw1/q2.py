"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje

I collaborated with the following classmates for this homework:
None
"""

from sklearn import datasets as ds
import pandas as pd
import matplotlib.pyplot as plot


def load_iris():
    """
    Loads the iris dataset from sklearn

    Returns
    -------
    A pandas DataFrame with the format: sepal length (cm),
    sepal width (cm), petal length (cm), petal length (cm),
    target
    """
    # Load the iris dataset and convert it to a pandas DataFrame
    iris_dataset = ds.load_iris()
    iris_dataframe = pd.DataFrame.from_records(data=iris_dataset.data,
                                               columns=iris_dataset.feature_names)

    # Append the target value column to the DataFrame
    iris_dataframe['target'] = iris_dataset.target

    return iris_dataframe


def iris_boxplot(iris_info):
    """
    Make boxplots for sepal length, sepal width, petal length, and petal width.

    Parameters
    ----------
    iris_info
    """

    # sepal length plot
    sepal_length_plot = iris_info.boxplot(column=["sepal length (cm)"],
                                          by=["target"])
    sepal_length_plot.plot()
    plot.show()

    # sepal width plot
    sepal_width_plot = iris_info.boxplot(column=["sepal width (cm)"],
                                         by=["target"])
    sepal_width_plot.plot()
    plot.show()

    # petal length plot
    petal_length_plot = iris_info.boxplot(column=["petal length (cm)"],
                                          by=["target"])
    petal_length_plot.plot()
    plot.show()

    # petal width plot
    petal_width_plot = iris_info.boxplot(column=["petal width (cm)"],
                                         by=["target"])
    petal_length_plot.plot()
    plot.show()


def iris_sepal_scatterplot(scatterplot_info):
    """
    Make a scatterplot for sepal with the length on the x-axis
    and the width on the y-axis, with each species type colored a different color.

    Parameters
    ----------
    scatterplot_info
    """

    plot.figure()

    # Create a plot for sepal length and width, colored by species type.
    plot.scatter(scatterplot_info.get("sepal length (cm)"),
                 scatterplot_info.get("sepal width (cm)"),
                 c=scatterplot_info['target'])
    plot.title("Sepal Length and Width of Irises Grouped by Species")
    plot.xlabel("sepal length (cm)")
    plot.ylabel("sepal width (cm)")

    plot.show()


def iris_petal_scatterplot(scatterplot_info):
    """
    Make a scatterplot for petal with the length on the x-axis
    and the width on the y-axis, with each species type colored a different color.

    Parameters
    ----------
    scatterplot_info
    """

    # Create a plot for petal length and width, colored by species type.
    plot.figure()
    plot.scatter(scatterplot_info.get("petal length (cm)"),
                 scatterplot_info.get("petal width (cm)"),
                 c=scatterplot_info['target']
                 )
    plot.title("Petal Length and Width of Irises Grouped by Species")
    plot.xlabel("petal length (cm)")
    plot.ylabel("petal width (cm)")
    plot.show()


def main():
    # 2(a): Load the iris dataset
    iris_info = load_iris()

    # 2(b): Create boxplots
    iris_boxplot(iris_info)

    # 2(c): Create scatterplots
    iris_sepal_scatterplot(iris_info)
    iris_petal_scatterplot(iris_info)


if __name__ == "__main__":
    main()
