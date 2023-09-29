# 导入相关库
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from pathlib import Path


class YoLov5TRT(object):
    # 该函数可将img_dir文件夹下的img文件按照batch_size进行分批次
    def get_img_path_batches(self, batch_size, img_dir):
        ret = []
        batch = []
        for root, dirs, files in os.walk(img_dir):
            for name in files:
                if len(batch) == batch_size:
                    ret.append(batch)
                    batch = []
                batch.append(os.path.join(root, name))
        if len(batch) > 0:
            ret.append(batch)
        return ret

    # 该函数可将在img文件按照颜色为color，框的粗细为line_thickness,左上角和右下角的坐标为x（x的格式为[x1,y1,x2,y2]）将检测目标框起来
    # 并且在检测框旁边可将类别label标注出来
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (
                line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    # 设置参数：置信度阈值（默认为0.5），交并比阈值（默认为0.4）
    def __init__(self, engine_file_path, categories, conf_thresh=0.5, iou_threshold=0.4):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.categories = categories
        self.conf_thresh = conf_thresh
        self.iou_threshold = iou_threshold

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        batch_input_image = np.empty(shape=[3, self.input_h, self.input_w])
        image_raw = raw_image_generator
        # for i, image_raw in enumerate(raw_image_generator):
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
        batch_image_raw = image_raw
        batch_origin_h = origin_h
        batch_origin_w = origin_w
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output[0: (1) * 6001], batch_origin_h, batch_origin_w)
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            self.plot_one_box(
                box,
                batch_image_raw,
                label="{}:{:.2f}".format(
                    self.categories[int(result_classid[j])], result_scores[j]
                ),
            )
        use_time = end - start
        return batch_image_raw, use_time, result_boxes, result_scores, result_classid

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, image_raw):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        # image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=self.conf_thresh,
                                         nms_thres=self.iou_threshold)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


# 多线程加速
class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def infer(self, frame):
        batch_image_raw, use_time, result_boxes, result_scores, result_classid = self.yolov5_wrapper.infer(frame)
        return batch_image_raw, use_time, result_boxes, result_scores, result_classid


class yolov5_tensorrt_demo(object):
    def __init__(self, PLUGIN_LIBRARY, engine_file_path, categories):
        self.PLUGIN_LIBRARY = PLUGIN_LIBRARY
        self.engine_file_path = engine_file_path
        self.categories = categories
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.yolov5_wrapper = YoLov5TRT(engine_file_path, categories)
        self.thread1 = inferThread(self.yolov5_wrapper)
        self.thread1.start()
        self.thread1.join()
        print("load model OK!")

    def detect(self, frame, if_view=True):
        img, use_time, result_boxes, result_scores, result_classid = self.thread1.infer(frame)
        output = [use_time]
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            # 格式为[use_time,[id_1,score_1,x1_1,y1_1,x2_1,y2_2],[id_2,score_2,x1_2,y1_2,x2_2,y2_2]...]
            output.append(
                [int(result_classid[j]), float(result_scores[j]), int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        if if_view:
            cv2.imshow("show", img)
        return output

    def destroy(self):
        self.yolov5_wrapper.destroy()


if __name__ == "__main__":
    categories = ['3+2-2', '3jia2', 'aerbeisi', 'anmuxi', 'aoliao', 'asamu', 'baicha', 'baishikele', 'baishikele-2',
                  'baokuangli', 'binghongcha', 'bingqilinniunai', 'bingtangxueli', 'buding', 'chacui', 'chapai',
                  'chapai2', 'damaicha', 'daofandian1', 'daofandian2', 'daofandian3', 'daofandian4', 'dongpeng',
                  'dongpeng-b', 'fenda', 'gudasao', 'guolicheng', 'guolicheng2', 'haitai', 'haochidian', 'haoliyou',
                  'heweidao', 'heweidao2', 'heweidao3', 'hongniu', 'hongniu2', 'hongshaoniurou', 'jianjiao',
                  'jianlibao', 'jindian', 'kafei', 'kaomo_gali', 'kaomo_jiaoyan', 'kaomo_shaokao', 'kaomo_xiangcon',
                  'kebike', 'kele', 'kele-b', 'kele-b-2', 'laotansuancai', 'liaomian', 'libaojian', 'lingdukele',
                  'lingdukele-b', 'liziyuan', 'lujiaoxiang', 'lujikafei', 'luxiangniurou', 'maidong', 'mangguoxiaolao',
                  'meiniye', 'mengniu', 'mengniuzaocan', 'moliqingcha', 'nfc', 'niudufen', 'niunai', 'nongfushanquan',
                  'qingdaowangzi-1', 'qingdaowangzi-2', 'qinningshui', 'quchenshixiangcao', 'rancha-1', 'rancha-2',
                  'rousongbing', 'rusuanjunqishui', 'suanlafen', 'suanlaniurou', 'taipingshuda', 'tangdaren',
                  'tangdaren2', 'tangdaren3', 'ufo', 'ufo2', 'wanglaoji', 'wanglaoji-c', 'wangzainiunai', 'weic',
                  'weitanai', 'weitanai2', 'weitanaiditang', 'weitaningmeng', 'weitaningmeng-bottle', 'weiweidounai',
                  'wuhounaicha', 'wulongcha', 'xianglaniurou', 'xianguolao', 'xianxiayuban', 'xuebi', 'xuebi-b',
                  'xuebi2', 'yezhi', 'yibao', 'yida', 'yingyangkuaixian', 'yitengyuan', 'youlemei', 'yousuanru',
                  'youyanggudong', 'yuanqishui', 'zaocanmofang', 'zihaiguo']

    # 设置相对路径
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    yolov5 = yolov5_tensorrt_demo(PLUGIN_LIBRARY=ROOT / "build2/libmyplugins.so",
                                  engine_file_path=ROOT / "build2/yolov5s.engine",
                                  categories=categories)
    # 图片文件夹路径
    images_folder_path = ROOT / 'samples'
    total_time = 0
    i=1
    for filename in os.listdir(images_folder_path):
        image_file_path = os.path.join(images_folder_path, filename)
        frame = cv2.imread(image_file_path)
        results = yolov5.detect(frame)
        print(f'image{i}:   use_time:{results[0]:.2f}s')
        total_time += results[0]
        # 一张图片里的框序号从1开始
        k = 1
        for result in results[1:]:
            print(
                f'box{k}  id{k}:{result[0]}  name{k}:{categories[result[0]]}  score_{k}:{result[1]:.2f}  x1_{k}:{result[2]}  y1_{k}:{result[3]}  x2_{k}:{result[4]}  y2_{k}:{result[5]}')
            k += 1
    i += 1
    print(f'total_time:{total_time:.2f}')
    cv2.destroyAllWindows()
    yolov5.destroy()
