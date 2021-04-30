import typer
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from os import path


def main(tn: int, fn: int, tp: int, fp: int, outdir: str="./out/", title: str="", show: bool=True):
    """
    Creates a confusion matrix with TP, FP, TN, FN values in the outdir.
    Set title for plot title.
    """
    data = pd.DataFrame([[tn, fp],[fn, tp]], range(2), range(2)).to_numpy()

    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in data.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in data.flatten()/np.sum(data)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sn.set(font_scale=1.3)
    sn.heatmap(data, annot=labels, fmt="", cmap="Greys", cbar=False, xticklabels=True, yticklabels=True)

    plt.xlabel("Prediction")
    plt.ylabel("True")
    if title != "":
        plt.title(title, loc="right")
    if show:
        plt.show()

    if title != "" and not path.exists(outdir + f"{title}.png"):
        filename = f"{title}.png"
    else:
        i = 0
        while path.exists(outdir + "confusionMatrix-%s.png" % i):
            i += 1
        filename = "confusionMatrix-%s.png" % i

    plt.savefig(outdir + filename, dpi=400)


if __name__ == "__main__":
    typer.run(main)