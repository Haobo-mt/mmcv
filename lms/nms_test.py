import torch
import torchvision
from torchvision.ops import nms as t_nms

import torch_musa

import mmcv
from mmcv.ops import nms

import os


device = torch.musa.set_device(1)

base = "/data/lms/mmcv/lms"

bboxes = torch.load(os.path.join(base, "bboxes.pt")).musa()
scores = torch.load(os.path.join(base, "scores.pt")).musa()
iou_threshold = torch.load(os.path.join(base, "iou_threshold.pt")).musa()
offset = torch.load(os.path.join(base, "offset.pt")).musa()
max_num = torch.load(os.path.join(base, "max_num.pt")).musa()



print(f"LMS: offset: {offset}, max_num: {max_num}")


dets, inds = nms(bboxes, scores, iou_threshold=iou_threshold.item(
), offset=offset.item(), max_num=max_num.item())


print(dets)


t_res = t_nms(bboxes, scores, iou_threshold=iou_threshold.item())

print(torch.allclose(inds,t_res))


cpu_bboxes = bboxes.cpu()
cpu_scores = scores.cpu()
cpu_dets, cpu_inds = nms(cpu_bboxes, cpu_scores, iou_threshold=iou_threshold.cpu().item(
), offset=offset.item(), max_num=max_num.cpu().item())

print(torch.allclose(cpu_inds,t_res.cpu()))


