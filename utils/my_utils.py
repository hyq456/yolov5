import os.path

import matplotlib
from matplotlib import pyplot as plt
import cv2
import torch

from utils.metrics import box_iou
import numpy as np
import math
#保存分类错误图片
def save_classfy_error(detections,labels,img_path,save_path):
    iouv = 0.5
    error = torch.zeros(detections.shape[0], 10, dtype=torch.bool, device=detections.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv) & (labels[:, 0:1] != detections[:, 5]))  # iou大于阈值但分类错误
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(detections.device)
        error[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        im = cv2.imread(str(img_path))
        for i,d in enumerate(matches):
            t = labels[int(d[0].item())]
            p = detections[int(d[1].item())]
            t_class = t[0].item()
            p_class = p[-1].item()
            x1, y1, x2, y2 = p[:4].tolist()
            error_crop = im[math.ceil(y1):math.ceil((y2)), math.ceil(x1):math.ceil(x2)]
            _,file_name = os.path.split(img_path)
            cv2.imwrite(f'{save_path}\{file_name}-true-{t_class}pred-{p_class}-{i}.jpg', error_crop)
            print(f'{save_path}/{file_name}-true-{t_class}pred-{p_class}-{i}.jpg')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    cv2.imshow('aa',inp)
    cv2.waitKey(0)
    # plt.imshow(inp)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

def YOCO(images, aug, h, w):
    images = torch.cat((aug(images[:, :, :, 0:int(w/2)]), aug(images[:, :, :, int(w/2):w])), dim=3) if \
    torch.rand(1) > 0.5 else torch.cat((aug(images[:, :, 0:int(h/2), :]), aug(images[:, :, int(h/2):h, :])), dim=2)
    return images