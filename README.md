# confusion-matrix-plotter

Simple python script to create to confusion matrices for binary classification problems.

```
Usage: main.py [OPTIONS] TN FN TP FP

  Creates a confusion matrix with TP, FP, TN, FN values in the outdir. Set
  title for plot title.

Arguments:
  TN  [required]
  FN  [required]
  TP  [required]
  FP  [required]

Options:
  --outdir TEXT                   [default: ./out/]
  --title TEXT                    [default: ]
  --show / --no-show              [default: True]
  --help                          Show this message and exit.
```

## Example Matrix

<img src="./image/exampleMatrix.png" alt="drawing" width="600"/>