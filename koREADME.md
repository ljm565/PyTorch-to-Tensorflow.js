# PyTorch to TensorFlow.js
## Requirements
* [requirements.txt](requirements.txt)에 나와있습니다.
* timm 라이브러리는 아래 명령어로 가장 최신 버전을 설치하면 됩니다.
    ```
    pip install git+https://github.com/rwightman/pytorch-image-models.git
    ```
<br><br>

## Converting Process
* PyTorch - ONNX - TensorFlow - TensorFlow.js
<br><br>

## Usage
* File Tree<br>
    PyTorch 학습된 모델을 가지고 있어야 실행이 가능하고, "model" 폴더 내에 아래와 같이 파일 트리가 구성 되어야합니다.
    ```
    pt2tfjs
    ┣ model
    ┃ ┣ {ModelName}
    ┃   ┣ {ModelName}.pt
    ┗   ┗ {ModelName}.json
    ```
* Execution
    ```
    python3 pt2tfjs.py -n {saved_model_folder}
    ```
<br><br>

## Acknowledgement
* models/pplcnet.py 코드는 뒤에 링크된 코드에서 약간 변형하였습니다([PP-LCNet](https://github.com/ngnquan/PP-LCNet/blob/main/pplcnet.py)).