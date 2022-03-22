import time
# import cv2
import numpy as np
from .fixed_base import FixedBase
from util import InferErr, pre_extract, LoadImages


class BodyFixedTrt(FixedBase):
    # input_names = ["data"]
    # output_names = ["fc1"]

    def __init__(self, engine_file_path: str, onnx_file: str, gpu_id=0, using_half=True):
        super().__init__(engine_file_path, onnx_file, gpu_id=gpu_id, using_half=using_half)

    def __call__(self, inp):
        self.cuda_ctx.push()
        try:
            assert inp.flags['C_CONTIGUOUS'], 'input is not C_CONTIGUOUS'
            if inp.dtype != np.float32:
                inp = inp.astype(np.float32, copy=False)
            inf_list = [inp]  # 模型可能有多个输入所以用list装起来
            outputs = self.do_inference(inf_list)
            return outputs
        except Exception as e:
            raise InferErr(e)
        finally:
            self.cuda_ctx.pop()

    @classmethod
    def postprocess(cls, outputs):
        feature = outputs[0]
        # feature = feature[:512*batch]  # size of output in h&d is maximum, so it need slice real size.
        # In this case, input shape is fixed so output is full
        feature = feature.reshape((1, 512))
        return feature


def main_body_fixed(source_dir='images',
                 onnx_file='onnx_trt/body_yolov5_fixed_640.onnx',
                 engine_file='onnx_trt/body_yolov5_fixed.trt', repeat=1):
    net = BodyFixedTrt(engine_file, onnx_file, using_half=False)
    dataset = LoadImages(source_dir, img_size=640, stride=32)
    # for i in range(img_nums):
    for path, img, im0s, vid_cap in dataset:
        tic = time.time()
        img = np.float32(img) / 255.0
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img_batch = np.expand_dims(img, axis=0) # 增加维数ndim
        img_batch = img_batch.repeat(repeat, axis=0) # 增加该维度的shape
        img_batch = np.ascontiguousarray(img_batch)
    # for i in range(10):
    #     outputs = net(img_batch)

        outputs = net(img_batch)
        # feature = ExtractFeatureTrt.postprocess(outputs)
        # print('Net of extract feature forward time: {:.4f} seconds for p{:d}'.format((t1 - tic), i))
        print('Body detect time: {:.4f} seconds'.format(time.time() - tic))
    # features.append(feature)
    # return features


if __name__ == "__main__":
    img_path = "images/cropped_3_4.png"
    main_extract([img_path])