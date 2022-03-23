# x2onnx

Generate onnx from other framework, such as pytorch

## Structure

`yolov5`：core files generating from torch

`main.py`：the entry of main, including simplifying onnx

`lab.py`：infer with pytorch for yolov5

`convert.py`：convert the model using `torch.jit`

## Usage

download weights from [body_yolov5](https://download.csdn.net/download/wq_0708/85020977) to the directory `./weights`，then run

```bash
python main.py
```

