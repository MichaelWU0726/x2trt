# x2trt
Build TensorRT engine by parsing from other frameworks, such as onnx,pytorch,TensorFlow

[The other way by defining network](https://github.com/MichaelWU0726/TRTx)

## Structure

`libs`: core files to generate engine

`onnx_trt`：onnx and engine files

`util`：some common utils

`main.py`：the entry of main

## Usage

Place your onnx file to `./onnx_trt` firstly. Generate engine and infer by running:
```bash
python main.py
```

**Attention:** please modify `onnx_file` and `engine_file` values
