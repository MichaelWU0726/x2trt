import torch
import yaml
from . import Model, intersect_dicts


def load_model(weights='../../best_people.pt', hyp='data/hyp_scratch.yaml',cfg='models/yolov5_human.yaml'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(weights, map_location=device)
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    model = Model(cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'])
    model.load_state_dict(state_dict, strict=False)
    return model

def load_new_state(weights='../../state_dict.pt',hyp='data/hyp_scratch.yaml',cfg='models/yolov5_human.yaml'):
    # 必须使用cpu，否则最后一步torch.onnx.export会报错
    device = 'cpu'
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    model = Model(cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)

    # 读取1.7版pytorch权重。根据自己权重文件格式读取参数
    # checkpoint = torch.load(weights, map_location=device)
    # state_dict_key = 'state_dict'
    # if state_dict_key and state_dict_key in checkpoint:
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint[state_dict_key].items():
    #         name = k[7:] if k.startswith('module') else k
    #         new_state_dict[name] = v
    #     state_dict = new_state_dict
    # else:
    #     state_dict = checkpoint
    # model.load_state_dict(state_dict)
    #
    # weights_new = 'weights/body_yolov5.pth'
    # torch.save(model.state_dict(), weights_new, _use_new_zipfile_serialization=True)

    # 重新读取，导出对应的onnx
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.train(False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # model.load_state_dict(torch.load(weights, map_location=device))
    # model.eval()
    return model


def load_state(weights,hyp,cfg):
    device = 'cpu'
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    model = Model(cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)

    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
