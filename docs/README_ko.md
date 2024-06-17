# PyTorch to TensorFlow.js

## Introduction
이 레포지토리는 PyTorch 모델을 TensorFlow.js(TFJS)로 변환하는 코드를 제공합니다.
전반적인 과정은 `PyTorch -> ONNX -> TensorFlow -> TensorFlow.js` 순서로 모델을 변환합니다.
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
본 repository에서는 학습된 PyTorch 모델이 아래와 같은 파일 트리 구조의 outputs 폴더 안에 있음을 가정합니다.
```
pt2tfjs (repository)
├── src
└── outputs
    └── ${ModelName}
        ├── ${ModelName}.pt
        └── ${ModelName}.json

```
#### Execution
`-n` 옵션에는 `outputs` 폴더 안에 위치한 모델 폴더 이름만 넣어주면 됩니다.

```bash
python3 src/pt2tfjs.py -n ${ModelName}
```
<br>

## Acknowledgement
* models/pplcnet.py 코드는 뒤에 링크된 코드에서 약간 변형하였습니다([PP-LCNet](https://github.com/ngnquan/PP-LCNet/blob/main/pplcnet.py)).