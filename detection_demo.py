# 导入必要的包
import cv2
import os
import torch
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


class detection_demo:
    # 初始化函数把权重，推理图片，设备，图片大小，推理置信度阈值，推理交并比阈值，
    # 测试置信度阈值，测试交并比阈值，测试推理后的图片是否保存,推理后的图片是否观看，推理时是否使用数据增强
    def __init__(self, weights, detect_source, device, img_size=640, detect_conf_thres=0.25, detect_iou_thres=0.45,
                 if_save=False, if_view=False, augment=True):
        self.weights = weights
        self.source = detect_source
        self.device = device
        self.detect_conf_thres = detect_conf_thres
        self.detect_iou_thres = detect_iou_thres
        self.if_save = if_save
        self.if_view = if_view
        self.augment = augment
        # 将用f32数据权重导入模型
        self.model = attempt_load(self.weights, map_location=device)
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.num_classes = len(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def load_images(self):
        return LoadImages(self.source, img_size=self.img_size, stride=self.stride)

    def view_image(self,img,results):
        if self.if_view:
            results = results[1:]
            for result in results:
                label = f'{self.names[result[0]]} {result[1]:.2f}'
                plot_one_box((result[2],result[3],result[4],result[5]), img, label=label,
                             color=self.colors[result[0]],line_thickness=3)
            cv2.imshow("show", img)
            #显示5ms
            cv2.waitKey(1000)


    def save_image(self,img,results,save_path):
        if self.if_save:
            results = results[1:]
            for result in results:
                label = f'{self.names[result[0]]} {result[1]:.2f}'
                plot_one_box((result[2],result[3],result[4],result[5]), img, label=label,
                             color=self.colors[result[0]],line_thickness=3)
            cv2.imwrite(save_path, img)

    #参数为img输入图像，img0s为原始图像，两者差别为在LoadImages时会进行letterbox操作将图片大小进行改变
    def detect(self, *image):
        img, img0s = image
        # 设定设备
        set_logging()
        device = select_device(self.device)

        # 设备如果是cuda(gpu)就使用f16计算
        half = device.type != 'cpu'
        if half:
            self.model.half()

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(device).type_as(
                next(self.model.parameters())))  # run once

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        predict = self.model(img, augment=self.augment)[0]

        # Apply NMS
        predict = non_max_suppression(predict, self.detect_conf_thres, self.detect_iou_thres, agnostic=True)
        t2 = time_synchronized()

        # 一张图片的总时间
        use_time = t2 - t1
        output = [use_time]

        for _, det in enumerate(predict):
            p, s, im0, frame = path, '', img0s, getattr(dataset, 'frame', 0)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 解包，将两个角的坐标，置信度conf，类别cls解包出来
                for *xyxy, conf, cls in reversed(det):
                    id_ = int(cls)
                    score = conf
                    x1, y1, x2, y2 = xyxy
                    output.append([id_, float(score), int(x1), int(y1), int(x2), int(y2)])
        return output

    def map_calculation(self, jpg_folder_path, xml_folder_path):
        def mean_average_precision(pred_bboxes, true_boxes, iou_threshold, num_classes=self.num_classes):

            # pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]

            average_precisions = []  # 存储每一个类别的AP
            epsilon = 1e-6  # 防止分母为0

            # 对于每一个类别
            for c in range(num_classes):
                detections = []  # 存储预测为该类别的bbox
                ground_truths = []  # 存储本身就是该类别的bbox(GT)

                for detection in pred_bboxes:
                    if detection[1] == c:
                        detections.append(detection)

                for true_box in true_boxes:
                    if true_box[1] == c:
                        ground_truths.append(true_box)

                amount_bboxes = Counter(gt[0] for gt in ground_truths)

                for key, val in amount_bboxes.items():
                    amount_bboxes[key] = torch.zeros(val)
                # 此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}

                detections.sort(key=lambda x: x[2], reverse=True)

                # 初始化TP,FP
                TP = torch.zeros(len(detections))
                FP = torch.zeros(len(detections))

                total_true_bboxes = len(ground_truths)

                if total_true_bboxes == 0:
                    continue

                for detection_idx, detection in enumerate(detections):

                    ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

                    num_gts = len(ground_truth_img)

                    best_iou = 0
                    for idx, gt in enumerate(ground_truth_img):
                        # 计算当前预测框detection与它所在图片内的每一个真实框的IoU
                        iou = insert_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                    if best_iou > iou_threshold:
                        if amount_bboxes[detection[0]][best_gt_idx] == 0:
                            TP[detection_idx] = 1
                            amount_bboxes[detection[0]][best_gt_idx] = 1
                        else:
                            FP[detection_idx] = 1
                    else:
                        FP[detection_idx] = 1

                TP_cumsum = torch.cumsum(TP, dim=0)
                FP_cumsum = torch.cumsum(FP, dim=0)

                # 套公式
                recalls = TP_cumsum / (total_true_bboxes + epsilon)
                precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

                # 把[0,1]这个点加入其中
                precisions = torch.cat((torch.tensor([1]), precisions))
                recalls = torch.cat((torch.tensor([0]), recalls))
                # 使用trapz计算AP
                average_precisions.append(torch.trapz(precisions, recalls))

            return sum(average_precisions) / len(average_precisions)

        def insert_over_union(boxes_preds, boxes_labels):

            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]  # shape:[N,1]

            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

            x1 = torch.max(box1_x1, box2_x1)
            y1 = torch.max(box1_y1, box2_y1)
            x2 = torch.min(box1_x2, box2_x2)
            y2 = torch.min(box1_y2, box2_y2)

            # 计算交集区域面积
            intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

            box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
            box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

            return intersection / (box1_area + box2_area - intersection + 1e-6)

        # 所有预测的数据（目录里所有的照片）
        data = LoadImages(jpg_folder_path, img_size=self.img_size, stride=self.stride)
        # 预测的标注框数据
        pred_bboxes = []
        train_id = 0
        for path_, *image_, vid_cap_ in data:
            # 每张照片的所有预测的预测框结果[use_time,[id_1,score_1,x1_1,y1_1,x2_1,y2_1],[id_2,score_2,x1_2,y1_2,x2_2,y2_2]...]
            results_ = demo.detect(*image_)

            # 将结果转化为[train_idx,class_pred,prob_score,x1,y1,x2,y2]的格式
            # train_idx：指示图片编号，用于区分不同的图片
            # class_pred：预测的类别对应的id
            # prob_score：置信度
            # (x1, y1)：bbox左上角坐标（可能已经归一化）
            # (x2, y2)：bbox右下角坐标（可能已经归一化）

            # 将use_time删除
            results_ = results_[1:]
            for result_ in results_:
                # 在每个数据中的最前面添加图片编号，从1开始
                result_.insert(0, train_id)
                pred_bboxes.append(result_)
            train_id += 1

        # 真实的标注框数据与pred_boxes对应
        true_boxes = []
        train_id = 0
        # 获得xml文件里真实的标注数据
        # 获得xml文件对应的路径
        for filename in os.listdir(xml_folder_path):
            if filename.endswith('.xml'):
                xml_file_path = os.path.join(xml_folder_path, filename)
                # 获得xml文件里的数据
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                for member in root.findall('object'):
                    # 标注框的对应的类别
                    class_name = member[0].text
                    class_id = self.names.index(class_name)
                    # 左上角点和右下角点的坐标
                    x1 = int(member[4][0].text)
                    y1 = int(member[4][1].text)
                    x2 = int(member[4][2].text)
                    y2 = int(member[4][3].text)
                    score = 1
                    true_boxes.append([train_id, class_id, score, x1, y1, x2, y2])
            train_id += 1


        iou_thresholds = torch.arange(0.5, 1.0, 0.05)
        mAPs = []
        for iou_threshold in iou_thresholds:
            mAP = mean_average_precision(pred_bboxes, true_boxes, iou_threshold, self.num_classes)
            mAPs.append(mAP)
        mAP_50_95 = sum(mAPs) / len(mAPs)
        return mAP_50_95



#设置相对路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# 设置权重，推理图片，推理设备，推理时的图片大小默认为（640*640）   720*960大小经过实验最精确的推理大小为（672，896）
#（不一定输入模型的图片的大小相同，不一定和原始图片相同，会有check_size,letterbox处理）
# 推理的置信度阈值（默认为0.25），推理的交并比阈值（默认为0.45）
# 推理时图片是否保存（默认为否）,推理时图片是否观看（默认为否），推理时否使用数据增强（默认为是）

demo = detection_demo(weights=ROOT / 'runs/train/exp18/weights/best.pt',
                      detect_source=ROOT / 'data/images',
                      device='cuda',img_size=896,if_view =True,
                      )
# 获得名称和标注框的颜色
names = demo.names
# 获得推理的数据
dataset = demo.load_images()
# 图片的序号从1开始
total_time = 0
i = 1
for path, *image, vid_cap in dataset:
    results = demo.detect(*image)
    # 观察图片
    img, img0s = image
    demo.view_image(img0s, results)
    print(f'image{i}:   use_time:{results[0]:.2f}s')
    total_time += results[0]
    # 一张图片里的框序号从1开始
    k = 1
    for result in results[1:]:
        print(
            f'box{k}  id{k}:{result[0]}  name{k}:{names[result[0]]}  score_{k}:{result[1]:.2f}  x1_{k}:{result[2]}  y1_{k}:{result[3]}  x2_{k}:{result[4]}  y2_{k}:{result[5]}')
        k += 1
    i += 1
print(f'total_time:{total_time:.2f}')
cv2.destroyAllWindows()


# 用于计算map的jpg图像文件和xml标注数据
mAP = demo.map_calculation(jpg_folder_path=ROOT / 'data/jpg_folder_path',
                           xml_folder_path=ROOT / 'data/xml_folder_path')
mAP = float(mAP)
print(f'mAP = {mAP}')
