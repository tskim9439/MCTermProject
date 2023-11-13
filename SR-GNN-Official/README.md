# SR-GNN

## Paper data and code

This is the code for the AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). We have implemented our methods in both **Tensorflow** and **Pytorch**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html> or <https://www.kaggle.com/chadgostopp/recsys-challenge-2015>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup> or <https://competitions.codalab.org/competitions/11161>

There is a small dataset `sample` included in the folder `datasets/`, which can be used to test the correctness of the code.

We have also written a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper.

## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

Then you can run the file `pytorch_code/main.py` or `tensorflow_code/main.py` to train the model.

For example: `cd pytorch_code; python main.py --dataset=sample`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```

## Requirements

- Python 3
- PyTorch 0.4.0 or Tensorflow 1.9.0

## Other Implementation for Reference
There are other implementation available for reference:
- Implementation based on PaddlePaddle by Baidu [[Link]](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn)
- Implementation based on PyTorch Geometric [[Link]](https://github.com/RuihongQiu/SR-GNN_PyTorch-Geometric)
- Another implementation based on Tensorflow [[Link]](https://github.com/jimanvlad/SR-GNN)
- Yet another implementation based on Tensorflow [[Link]](https://github.com/loserChen/TensorFlow-In-Practice/tree/master/SRGNN)

## Citation

Please cite our paper if you use the code:

```
@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
```

## 실행 후 분석
기존 Sample 데이터로 학습을 진행 시, MRR 등 평가 수치가 1.x, 4.x 등으로 매우 낮은 값을 나타냅니다.
이는 논문에서 나온 수치와 다르게 매우 안 좋은 값이어서
논문에 나온 수치를 보기 위해선 Yoochoose/diginetica 데이터를 통해 실험을 진행해야 할 것 입니다.

## Preprocess Error 해결 방법
Yoochoose/diginetica 데이터를 다운로드 후 실험을 시도해보면, Preprocess 과정에서 에러가 발생합니다.
이는 실제 데이터의 key 값이 코드에서 나온 key 값과 달라서 나타납니다.
이를 해결하기 위해선 ['datasets/preprocess.py']('./datasets/preprocess.py') 파일을 수정해야 합니다.
현재 diginetica 에 대해선 수정을 진행했습니다.
40 번 줄에 아래의 코드를 추가해 주세요.

``` python
if opt.dataset == 'diginetica':
    data['session_id'] = data['sessionId']
    data['item_id'] = data['itemId']
```