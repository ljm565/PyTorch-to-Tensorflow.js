# PyTorch to TensorFlow.js
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
This repository provides code to convert PyTorch models to TensorFlow.js (TFJS).
The overall process involves converting the model in the following order: `PyTorch -> ONNX -> TensorFlow -> TensorFlow.js`.
<br><br>

## Requirements Installation
* Shown in [requirements.txt](requirements.txt)
* The most recent version of timm
```bash
# requirements.txt
pip install -r requirements.txt

# timm package install
pip install git+https://github.com/rwightman/pytorch-image-models.git
```
<br>

## Converting Process
* PyTorch - ONNX - TensorFlow - TensorFlow.js
<br><br>

## Usage
#### File Tree
Let's assume that the trained PyTorch model is inside the `outputs` folder with the following file tree structure:
```
pt2tfjs (repository)
├── src
└── outputs
    └── ${ModelName}
        ├── ${ModelName}.pt
        └── ${ModelName}.json

```
#### Execution
For the `-n` option, you only need to provide the name of the folder containing the model inside the `outputs` folder.
```bash
python3 src/pt2tfjs.py -n ${ModelName}
```
<br>

## Acknowledgement
* models/pplcnet.py code is slightly changed from [PP-LCNet](https://github.com/ngnquan/PP-LCNet/blob/main/pplcnet.py).