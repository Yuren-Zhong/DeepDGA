# DGA-generated Domain Detection

## Setup
```
conda install keras tensorflow-gpu
```

## Usage
### Training:
```
python train.py
```
Modify the `train.py` to choose different models; or modify the `models.py` to implement new models.
### Predicting:
```
python predict.py
```
Modify the `domain.json`  to add/remove the domains to be classified; or modify the `predict.py` to choose different models or weighs.
### Evaluating:
```
python evaluate_on_other_dga.py
```
Modify the `evaluate_on_other_dga.py` to choose different DGA set to evaluate.

## Dataset
Training on the dataset built by https://github.com/andrewaeva/DGA, including 1,000,000 legit domains and 801,667 DGA generated domains. While calling `dataset.load_data`, if specify `filter=True` then legal domains that end with different suffixes than DGA generated domains are not loaded, and also suffixes of all loaded domains are removed.

Evaluating on the dataset provided by http://data.netlab.360.com/feeds/dga/dga.txt, https://github.com/philarkwright/DGA-Detection, https://github.com/nickwallen/botnet-dga-classifier and https://github.com/ClickSecurity/data_hacking .