import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models
from skimage import measure, draw, data, util
from skimage.filters import threshold_otsu, threshold_local,threshold_minimum,threshold_mean,rank
from skimage.morphology import disk
import skimage

from utils import im_convert, get_device, check_accuracy
from datatest import Medical_Data_test
import argparse
import ttach as tta

# Specify the graphics card
torch.cuda.set_device(1)

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
test_path = './input/'
output_path = './output/'

def to_json(points,img_path):
    '''
    生成json文件
    '''
    json_out ={}
    floder_name = img_path.split("/")[-3]
    video_name = img_path.split("/")[-2]
    image_name = img_path.split("/")[-1]
    json_out["folderName"] = floder_name
    json_out["subfolderName"] = video_name
    json_out["imageFileName"] = image_name

    json_out["points"] = points

    if not Path(output_path+floder_name+"/"+video_name+"/").exists():
        os.makedirs(Path(output_path+floder_name+"/"+video_name+"/"))
    
    with open(output_path+floder_name+"/"+video_name+"/"+img_path.split("/")[-1].replace(".png",".json"),'w') as file_obj:
        json.dump(json_out,file_obj)

def generate_points(image_out, connectivity=2):
    # generate centre of mass
    label_img = measure.label(image_out, connectivity=2)
    props = measure.regionprops(label_img)
    # generate prediction points
    points = []
    areas = []
    bboxs = []
    for prop in props:
        # 这里注意x，y别搞反了,输入是 288x512,第零维度是y,第一维是x，
        point = {}
        point["y"] = prop.centroid[0]
        point["x"] = prop.centroid[1]
        bboxs.append(prop.bbox)
        points.append(point)
        areas.append(prop.area)
    return points, areas, bboxs

def OSTU(predict):
    radius = 2
    selem = disk(radius)
    threshold_global_otsu = threshold_otsu(predict)
    image_out = predict >= threshold_global_otsu
    # 开运算  圆形kernel
    kernel = skimage.morphology.disk(2)
    image_out =skimage.morphology.opening(image_out, kernel)
    # 生成点
    points, areas, bboxs = generate_points(image_out)
    points_big = []
    bboxs_big = []

    if len(areas) > 0:
        area_average = int(sum(areas)/len(areas))
        for i in range(len(areas)):
            if areas[i] > area_average:
                points_big.append(points[i])
                bboxs_big.append(bboxs[i])

    if len(points_big):
        img_out = check_out(image_out,points_big,bboxs_big)
        points, areas, bboxs = generate_points(img_out)
        return points
    else:
        return points

def check_out(image_out,points_big,bboxs_big):
    # print(image_out.shape[0],image_out.shape[1])
    for i in range(len(points_big)):
        x = int(points_big[i]["x"])
        y = int(points_big[i]["y"])
        ymin = bboxs_big[i][0]
        xmin = bboxs_big[i][1]
        ymax = bboxs_big[i][2]
        xmax = bboxs_big[i][3]
        if xmax-xmin<=ymax-ymin:
            # y don't change
            image_out[y,xmin:xmax] = False

        elif xmax-xmin>ymax-ymin:
            # x don't change
            image_out[ymin:ymax,x] = False

    return image_out

def predict(model_path, test_loader, iftta):

    model = torch.load(model_path, map_location='cuda:1')
    model = model.to(device)
    model.eval()

    if iftta:
        print("Using TTA")
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    for batch in tqdm(test_loader):
        imgs, imgs_path = batch
        imgs = imgs.to(device)

        with torch.no_grad():
            logits = torch.sigmoid(model(imgs))

        for i in range(len(imgs_path)):
            predict = im_convert(logits[i], False)
            points = OSTU(predict)
            to_json(points,imgs_path[i])


def main():
    # Specify command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                        choices=["landmark_detection", "domain_transformation"])

    batch_size = 16
    num_workers = 2
    model_path = './models/intra_Diceloss.pt'
    test_dataset = Medical_Data_test(test_path, data_mode='simulator', set_mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False
        )

    predict(model_path, test_loader, True)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()


