# 540 Term Project

## Dependencies

Dependencies that can be installed through PyPI are

* Tensorflow
* Keras
* NumPy
* SciPy
* scikit-image
* MatPlotLib
* tqdm

In addition, the following packages are also required

* [Pytorch](http://pytorch.org/)
* [PyJet](https://abhmul.github.io/PyJet/)

## Setting up the data

Place the testing data into a folder on the top level of the repository called `input/test` and the training data in `input/train` in the same format as Kaggle's Data Science Bowl competition's datasets.

## Creating the submission

Run the command from the `src` directory

```bash
python3 run_model.py train-unet-1 --test
```

to generate a submission. The submission can be found in the `submissions` folder under the name `train-unet-1.csv`
