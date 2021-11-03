import cv2

#保存分类错误图片

def save_classfy_error(pred_class,pbox,labels,img_path,save_path):
    for i , p in enumerate(pred_class):
        im = cv2.imread(img_path)
        p_class = p.item() #预测类别
        t_class = labels[i].item()  #真实类别
        # if p_class == 0:
        #     p_class = 1
        # elif p_class == 1:
        #     p_class = 3
        # elif p_class == 2:
        #     p_class = 0
        # elif p_class == 3:
        #     p_class = 2
        if p_class != t_class:
            x1,y1,x2,y2 = pbox[i].tolist()
            error_crop = im[y1:y2,x1:x2]
            cv2.imwrite(f'{save_path}/true:{t_class}pred:{p_class}',error_crop)
