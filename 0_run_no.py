import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import glob
import pickle

# 0 ~ 1로 픽셀값 뽑아내기!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

hist_list = []
count = 0
def draw_histogream(source, img, bbox, identities=None): #이미지, 좌표
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        label = '{}{:d}'.format("", id)
        if label == '1': #1번 사람인경우
            his_r = np.zeros(256)
            his_g = np.zeros(256)
            his_b = np.zeros(256)
            step_x = max(int((x2-x1)/256),1)
            step_y = max(int((y2-y1)/256),1)
            #print(step_x,step_y,x1,y1,x2,y2)
            for idx_x in range(x1,x2,step_x):
                for idx_y in range(y1,y2,step_y):
                    pix = img[idx_y,idx_x] #BGR
#                    img[idx_y,idx_x-100]=pix #테스트용
                    ##############################################
                    # 빈도수 세기
                    his_b[pix[0]] = his_b[pix[0]] + 1
                    his_g[pix[1]] = his_g[pix[1]] + 1
                    his_r[pix[2]] = his_r[pix[2]] + 1
            #최대값이 1이 되게 정규화
            his_b = his_b / np.max(his_b)
            his_g = his_g / np.max(his_g)
            his_r = his_r / np.max(his_r)
            #프레임당 하나의 배열로 합침
            frame_rgb = np.concatenate((his_b,his_g,his_r))
            #print(x1, y1, x2, y2, frame_rgb)
            #print(frame_rgb.shape,hist_np.shape)
            return frame_rgb
        return None


def init(opt):
    yolo_weights, deep_sort_weights, imgsz = \
        opt.yolo_weights, opt.deep_sort_weights, opt.img_size
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
#    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    # Check if environment supports image displays
#        show_vid = check_imshow()
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return imgsz, device, half, model

def detect(opt, source, imgsz, device, half, model):
    print("\n"+source, end='')
    dataset = LoadImages(source, img_size=imgsz)
    hist_np = np.zeros(256*3)
    start = time.time()
    
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
    ###########################################
    # 반복 부분
    ###########################################
    object_class_ids = []
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                # *xyxy, conf, cls
                # object_boxes.append([int(xy[0]+xy[2])/2, int(xy[1]+xy[3])/2, int(xy[2]-xy[0]), int(xy[3]-xy[1])]) #중심, 가로세로 크기
                # conf 정확도
                # cls 클래스 번호
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    object_class_ids.append(cls)#객체 종류 번호

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    frame_rgb = draw_histogream(source, im0, bbox_xyxy, identities) #출력
                    if frame_rgb is not None:
                        #print(frame_rgb.shape[0])
                        #print(hist_np.shape[0])
                        if hist_np.shape[0]%10==0:
                            print("|", end='')
                            end = time.time()
                            print(end-start,"sec")
                    
                        hist_np = np.block([[hist_np],[frame_rgb]])
                        # 프레임이 256이면 출력해보자-->쓰레드문제로 파일로 저장만 올바르게됨
                        if hist_np.shape[0] == 32:
                            #plt.matshow(hist_np)
                            #plt.savefig(source+'.png')
                            
                            with open(source+'.pkl', 'wb') as f:
                                pickle.dump(hist_np, f)
                                return
                    #draw_boxes(im0, bbox_xyxy, identities) #화면출력
            else:
                deepsort.increment_ages()

            # Stream results
            # cv2.imshow("test", im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
    for i in range(len(object_class_ids)):
        print(object_class_ids[i])

def GetInfo(args, target_dir, imgsz, device, half, model):
    bigdata_files = glob.glob(target_dir)

    for idx1 in range(len(bigdata_files)):
        detect(args, bigdata_files[idx1], imgsz, device, half, model)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='2.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        imgsz, device, half, model = init(args)
        GetInfo(args, "./data_no_margin/*/*.mp4", imgsz, device, half, model) #이 함수를 아래에 반복 추가 함으로써 디렉토리 별로 순차적으로 진행가능


