import torch
import yaml
from yolov5.utils import LoadImages
from yolov5.utils import non_max_suppression
# from yolov5.models import attempt_load
from yolov5 import load_state

# for x in ['s', 'm', 'l', 'x']:
#     attempt_download(f'yolov5{x}.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = 'weights/state_dict.pt'
hyp = 'yolov5/config/hyp_scratch.yaml'
cfg = 'yolov5/config/yolov5_human.yaml'
# ckpt = torch.load(weights, map_location=device)
# with open(hyp) as f:
#     hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
# model = attempt_load(weights, map_location=device)
model = load_state(weights, hyp, cfg).to(device)
stride = int(model.stride.max())
source = 'images'
dataset = LoadImages(source, img_size=640, stride=stride)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    print(pred)
