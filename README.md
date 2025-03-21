# Road Surface Classification with a focus on Gravel detection
This repository presents a series of scripts that aid in the reproduction of well-behaved models for classifying gravel on road surfaces, but also present a high accuracy in a variety of other road surfaces.
We've split the training in each stage and each model to a single script run, as to aid in the monitoring of the results and allow the adjustment of the parameters in case of inconsistent results or any other potential problems.

For more information on the results of our procedures, consider reading our paper: [Advanced driving assistance integration in electric motorcycles: road surface classification with a focus on gravel detection using deep learning](https://doi.org/10.3389/frai.2025.1520557)

## Requirements
It is necessary to install Python 3, TensorFlow, Numpy, Keras 3, and sci-kit learn.
The dataset is provided [here](https://figshare.com/ndownloader/files/36625041). Make sure to place the data in a subdirectory of the project root called "rscd".

## Usage
To train the ConvNeXt-T model, simply run:
```
python3 convnexttiny.py
```


After this, it can be fine-tuned using focal loss for 8 epochs using:
```
python3 improve.py convnexttiny_gravel.keras
```

It will then create a convnexttiny_improved.keras file, which contains the models weights, and can then be tested using:
```
python3 test.py convnexttiny_improved.keras
```


This will create the results of such tests, where the "epitaph" variable is the confusion matrix.
We've also included all results from our models in the "results" subdirectory of this repository, where it can be used for further analysis if necessary.

These results can then be visualized by the scripts provided in the "plots" subdirectory.
For instance, to plot the confusion matrix present in the results/convnexttiny_improved.json file, we must run:
```
cd plots
python3 epitaph.py subdirs.json ../results/convnexttiny_improved.json
```

Similarly, to plot the training logs for all the models, we must do:
```
cd plots
python3 train_res.py ../results
```

Note that all the plots have already been provided, in PDF format, for future comparison.
