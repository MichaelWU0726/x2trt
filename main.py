from libs import main_helmetfire_fixed


if __name__ == "__main__":
    engine_file = 'onnx_trt/body_yolov5_fixed_640.trt'
    onnx_file = 'onnx_trt/body_yolov5_fixed_640.onnx'
    main_helmetfire_fixed(engine_file=engine_file, onnx_file=onnx_file)