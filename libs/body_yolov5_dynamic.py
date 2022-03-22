import time
import numpy as np
from .dynamic_base import TensorrtBase
from util import HostDeviceMem, InferErr, LoadImages
import tensorrt as trt
import pycuda.driver as cuda


class BodyTrt(TensorrtBase):
    input_names = ["input_img"]
    output_names = ["out"]

    def __init__(self, engine_file_path: str, out_num: int, inp_shape=(512, 512), gpu_id=0):
        super().__init__(engine_file_path, gpu_id=gpu_id)
        self.out_shape = out_num
        # self.inp_shape = inp_shape
        self.buffers = self._allocate_buffer(inp_shape)

    def __call__(self, inp, batch_size):
        self.cuda_ctx.push()
        try:
            assert inp.flags['C_CONTIGUOUS'], 'input is not C_CONTIGUOUS'
            if inp.dtype != np.float32:
                inp = inp.astype(np.float32, copy=False)
            inf_in_list = [inp]
            binding_shape_map = {'input_img':inp.shape}
            outputs = self.do_inference(inf_in_list, binding_shape_map=binding_shape_map, batch_size=batch_size)
            return outputs
        except Exception as e:
            raise InferErr(e)
        finally:
            self.cuda_ctx.pop()

    def _allocate_buffer(self, inp_shape: tuple):
        """Allocate buffer when output shape is dynamic. Also for (h,w) is dynamic and change params inp_shape
        :inp_shape: dynamic shape for normally expand the buffer size. It equal to h * w
        """
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()

        for binding in self.binding_names:
            binding_idx = self.engine[binding]
            if binding_idx == -1:
                print("Error Binding Names!")
                continue

            dims = self.engine.get_binding_shape(binding)
            # print('binding shape: ', dims)
            if dims[-1] == -1:
                # assert (input_shape is not None)
                dims[-2], dims[-1] = inp_shape
            if not self.engine.binding_is_input(binding) and dims[-2] == -1:
                # assert (output_shape is not None)
                dims[-2] = self.out_shape
            # print('binding shape is changed to: ', dims)
            # trt.volume() return negtive volue if -1 in shape
            size = abs(trt.volume(dims)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # if self.engine.binding_is_input(binding):
            #     print('input size: ', size)
            #     print('input dtype: ', dtype)
            # else:
            #     print('output size: ', size)
            #     print('output dtype: ', dtype)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings[binding_idx] = int(device_mem)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def postprocess(self, outputs, batch: int):
        out_size = batch * self.out_shape
        loc_shape = (batch, self.out_shape, 4)
        conf_shape = (batch, self.out_shape, 2)
        landms_shape = (batch, self.out_shape, 10)
        loc, landms, conf = outputs
        # print('out size: ', loc.shape)
        # print('loc size: ', out_size * 4)
        # size of output in h&d is maximum, so it need slice real size.
        loc_real = loc[:4 * out_size]
        landms_real = landms[:10 * out_size]
        conf_real = conf[:2 * out_size]
        loc = loc_real.reshape(loc_shape)/box_scale
        landms = landms_real.reshape(landms_shape)/box_scale
        conf = conf_real.reshape(conf_shape)/box_scale
        return loc, conf, landms


def main_body_dynamic(source_dir='images',
                engine_file='onnx_trt/face_R50_batch_shape_512.trt',
                onnx_file='onnx_trt/face_r50_batch_shape.onnx',
                repeat=1, build_flag=False, batch_size=10):
    stride = [8,16,32]
    box_nums = {512: 10752, 640: 16800, 1024: 43008}

    if build_flag:
        print("===> build tensorrt engine...")
        dynamic_shapes = {"input_img": ((1, 3, 256, 256), (1, 3, 640, 640), (10, 3, 1024, 1024))}
        BodyTrt.build_engine(
            onnx_file_path=onnx_file, engine_file_path=engine_file,
            dynamic_shapes=dynamic_shapes,
            max_batch_size=batch_size)
    else:  # load engine to do
        print("===> load tensorrt engine...")
        # pass

    dataset = LoadImages(source_dir, img_size=640, stride=32)
    for path, img, im0s, vid_cap in dataset:
        img /= 255.0
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img_batch = np.expand_dims(img, axis=0)
        img_batch = img_batch.repeat(repeat, axis=0)
        img_batch = np.ascontiguousarray(img_batch)

    net = BodyTrt(engine_file, 15120, inp_shape=(640,640))
    tic = time.time()
    outputs = net(img_batch, repeat)
    t1 = time.time()
    # locs, confs, landms = net.postprocess(outputs, repeat)
    # t2 = time.time()
    print('Net of Retina forward time: {:.4f} seconds'.format((t1 - tic)))
    # print('infer {} pictures time: {:.4f} seconds'.format(repeat, (t2 - tic)))


if __name__ == '__main__':
    img_path = "images/test_3.png"
    main_body(img_path, repeat=10)
