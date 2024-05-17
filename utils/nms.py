import torch
from utils.iou import bbox_iou, box_convert
import torchvision

torch.set_printoptions(precision=4, sci_mode=False)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, max_nms=300):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    max_nms = max_nms

    method = 'merge'  #

    output = [torch.zeros((0, 6), device=prediction.device)] * len(prediction)

    for image_i, pred in enumerate(prediction):

        # ------------ 提取 满足阈值的 行 --------------
        pred = pred[pred[:, 4] > conf_thres]

        #  ------------ 提取 满足 wh 的 行 --------------
        pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]

        # sort by confidence and remove excess boxes
        pred = pred[pred[:, 4].argsort(descending=True)[:max_nms]]

        # 没有目标就去下一个尺度
        n = pred.shape[0]  # number of boxes
        if not n:
            continue

        # conf = obj_conf * cls_conf
        pred[:, 5:] *= pred[:, 4:5]

        # tx,ty,tw,th -> x1,y1,x2,y2
        box = box_convert(pred[:, :4], in_fmt='cxcywh', out_fmt='xyxy')

        # ----------- 选取 概率最大的类别 -------------
        conf, j = pred[:, 5:].max(1, keepdim=True)

        # x1, y1, x2, y2, confidence, class
        pred = torch.cat((box, conf, j.float()), 1)

        boxes, scores = pred[:, :4], pred[:, 4]

        # Each index value correspond to a category
        i = torchvision.ops.batched_nms(boxes, scores, j.squeeze(1), iou_thres)
        if method == 'merge' and 1 < n < 3E3:  # Merge NMS (boxes merged using weighted mean)
            # 目标个数 < 矩阵运算接受最大 长宽
            iou = bbox_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]  # box weights
            pred[i, :4] = torch.mm(weights, pred[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        elif method == 'fast':  # FastNMS from https://github.com/dbolya/yolact
            iou = bbox_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
            i = iou.max(0)[0] < iou_thres
        output[image_i] = pred[i]
    return output
