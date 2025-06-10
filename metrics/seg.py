import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel

from utils.registry import METRIC_REGISTRY
import warnings
from sklearn.metrics import confusion_matrix

from PIL import Image

# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('./Fusion_results1/Pred_seg_' + weight_name + '_' + image_name[i] + '.png')


def compute_results(conf_total):
    n_class = conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1

    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return np.nanmean(iou_per_class)  # Return mIoU directly from this function

@METRIC_REGISTRY.register()
def calculate_miou(seg_preds, labels, n_classes=9):
    conf_total = np.zeros((n_classes, n_classes))

    for prediction, label in zip(seg_preds, labels):
        conf = confusion_matrix(y_true=label.flatten(), y_pred=prediction.flatten(), labels=list(range(n_classes)))
        conf_total += conf

    miou = compute_results(conf_total)
    return miou

