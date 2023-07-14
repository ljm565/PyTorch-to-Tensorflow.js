# PyTorch to TensorFlow.js
[README.md 한국어 버전](koREADME.md)
<br><br>

## Requirements
* Shown in [requirements.txt](requirements.txt)
* The most recent version of timm
    ```
    pip install git+https://github.com/rwightman/pytorch-image-models.git
    ```
<br>

## Converting Process
* PyTorch - ONNX - TensorFlow - TensorFlow.js
<br><br>

## Usage
* File Tree<br>
    You have to have trained torch model with the below file tree in the "model" named folder.
    ```
    pt2tfjs
    ┣ model
    ┃ ┗ {ModelName}
    ┃   ┣ {ModelName}.pt
    ┗   ┗ {ModelName}.json
    ```
* Execution
    ```
    python3 pt2tfjs.py -n {saved_model_folder}
    ```
<br>

## Acknowledgement
* models/pplcnet.py code is slightly changed from [PP-LCNet](https://github.com/ngnquan/PP-LCNet/blob/main/pplcnet.py).