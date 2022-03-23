from yolov5 import load_state
import torch.backends.cudnn as cudnn
from yolov5 import export_onnx
import onnx


# def file_size(path):
#     # Return file/dir size (MB)
#     path = Path(path)
#     if path.is_file():
#         return path.stat().st_size / 1E6
#     elif path.is_dir():
#         return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
#     else:
#         return 0.0

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4'
        )
        return
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model

if __name__ == "__main__":
    simplify = True
    dynamic = False
    weights = 'weights/body_yolov5.pth'
    hyp = 'yolov5/config/hyp_scratch.yaml'
    cfg = 'yolov5/config/yolov5_human.yaml'
    model = load_state(weights=weights, hyp=hyp, cfg=cfg)
    cudnn.benchmark = True
    onnx_path = 'onnx/body_yolov5_fixed_640.onnx'
    # input_name = ['input']
    # output_name = ['output']
    # input = torch.Tensor(torch.randn(1, 3, 640, 640)).cuda()
    # torch.onnx.export(net, input, 'face_resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True, opset_version=11)
    image_shape = (640, 640)
    dynamic_ax = {'input': {0: 'batch', 2: 'image_height', 3: 'image_wdith'},
                  'out_1': [0, 1],
                  'out_2': [0,2,3],
                  'out_3': [0,2,3],
                  'out_4': [0,2,3]}
    export_onnx(model, image_shape, onnx_path, dynamic_onnx=dynamic, device='cpu')
    # except Exception as e:
    #     print("***********Error:**********")
    #     print(e)

    print("========check onnx========")
    test = onnx.load(onnx_path)
    onnx.checker.check_model(test)
    print("==> Passed")

    prefix = 'ONNX: '
    if simplify:
        try:
            # check_requirements(['onnx-simplifier'])
            import onnxsim

            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                test,
                dynamic_input_shape=False,
                input_shapes={'input': list(image_shape)} if dynamic else None
            )
            assert check, 'assert check failed'
            model_onnx = remove_initializer_from_input(model_onnx)
            onnx.save(model_onnx, onnx_path)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')
        # print(f'{prefix} export success, saved as {onnx_path} ({file_size(f):.1f} MB)')
        print(f'{prefix} export success, saved as {onnx_path}')
