import os.path

import cv2
import torch
from utils.metrics import box_iou
import numpy as np
import math

#保存分类错误图片
def save_classfy_error(detections,labels,img_path,save_path):
    iouv = 0.5
    # for i , p in enumerate(pred_class):
    #     im = cv2.imread(img_path)
    #     p_class = p.item() #预测类别
    #     t_class = labels[i].item()  #真实类别
    #     # if p_class == 0:
    #     #     p_class = 1
    #     # elif p_class == 1:
    #     #     p_class = 3
    #     # elif p_class == 2:
    #     #     p_class = 0
    #     # elif p_class == 3:
    #     #     p_class = 2
    #     if p_class != t_class:
    #         x1,y1,x2,y2 = pbox[i].tolist()
    #         error_crop = im[y1:y2,x1:x2]
    #         cv2.imwrite(f'{save_path}/true:{t_class}pred:{p_class}',error_crop)
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
