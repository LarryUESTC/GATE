# GATE


## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Testing](#test)
0. [Training](#train)

## Installation
pip install -r requirements.txt 

## Preparation

Pls see Configs >>>[here](./main.py/) Lines 528-544<<<.

| Configs          | Custom key | note   |
|------|------|------|
| dataset               | ABIDE/FTD       | dataset|
| seed                  | random seeds     | default is 2022|
| train                 | train or test (bool) | None|
| label_rate            | 0.2 (flaot)        | None|
| knn                   | the top k in KNN (int)   | default is 5|
| SAMA                  | use SA+MA (bool)   | default is True|
| beta                  |  the trade-off coefficient in Eq. (2)     | default is 0.2|
| lastdim               | dimensions in SSL |  256 and 512|

other args:
* `--lastdim2` dimension in fine-tune stage
* `--epoch_all1` Epoch in SSL
* `--epoch_all2` Epoch in fine-tune stage
* `--lr1` learning rate in SSL
* `--lr2` learning rate in fine-tune stage
* `--weight_decay1` weight decay in SSL
* `--weight_decay2` weight decay in fine-tune stage

## Testing

* Classification Task ( Default dataset is ABIDE ) 
```shell
python main.py --train False
```
 
## Training

* Classification Task ( Default dataset is ABIDE )
```shell
python main.py --train True
```
