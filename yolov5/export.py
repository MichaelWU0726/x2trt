import torch
import numpy as np

def export_onnx(model, image_shape, onnx_path, half=False, dynamic_onnx=True, dynamic_axes=None, op_version=11, batch_size=1, device='cuda'):
    # operator_export_type = torch._C._onnx.OperatorExportTypes.ONNX
    x, y = image_shape
    if half:
        inp = np.random.randn(batch_size, 3, x, y).astype(np.float16)
    else:
        inp = np.random.randn(batch_size, 3, x, y).astype(np.float32)
    if device == 'cuda':
        device = torch.device('cuda')
        inp = torch.from_numpy(inp).to(device)
    else:
        inp = torch.from_numpy(inp).to('cpu')
    if dynamic_onnx:
        assert dynamic_axes is not None, 'Please provide dynamic_axes'
        # dynamic_ax = {'input_img': {0: 'batch', 2: 'image_height', 3: 'image_wdith'},
        #               'out_1': [0, 1]}
        torch.onnx.export(model, inp, onnx_path,
                          input_names=list(dynamic_axes.keys())[:1], output_names=list(dynamic_axes.keys())[1:], verbose=False, opset_version=op_version,
                          dynamic_axes=dynamic_axes)
    else:
        torch.onnx.export(model, inp, onnx_path,
                          input_names=["input_img"], output_names=["out_1","out_2","out_3","out_4"], verbose=False, opset_version=op_version)