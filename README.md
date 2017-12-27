# DGA-generated Domain Detection

## Setup
```
conda install keras tensorflow-gpu
```

## Usage
###Training:
```
python train.py
```
One may modify the `train.py` to choose different models or modify the `models.py` to implement new models.
###Predicting:
1) modify the `domain.json`  to add the domains to be classified.
2)
```
python predict.py
```
One may modify the predict.py to choose different models or weighs
## Dataset
Use the dataset built by https://github.com/andrewaeva/DGA, including 1,000,000 legit domains and 801,667 DGA generated domains.