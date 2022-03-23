import torch
import yaml

from torch2onnx.models import Model


def convert_model(model, input=torch.tensor(torch.rand(size=(1, 3, 640, 640)))):
    model = torch.jit.trace(model, input)
    torch.jit.save(model, 'weights/model.tjm')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg='config/yolov5_human.yaml'
hyp='config/hyp_scratch.yaml'
with open(hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)
model = Model(cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)
convert_model(model)