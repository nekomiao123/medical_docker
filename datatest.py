import json
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.stats import multivariate_normal

def normalize(image):
    _range = np.max(image) - np.min(image)
    img = (image - np.min(image)) / _range
    return img

def points_to_gaussian_heatmap(centers, height, width, scale):
    if centers:
        gaussians = []
        for x,y in centers:
            s = np.eye(2)*scale
            g = multivariate_normal(mean=(x,y), cov=s)
            gaussians.append(g)

        # create a grid of (x,y) coordinates at which to evaluate the kernels
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x,y)
        xxyy = np.stack([xx.ravel(), yy.ravel()]).T

        # evaluate kernels at grid points
        zz = sum(g.pdf(xxyy) for g in gaussians)
        img = zz.reshape((height,width))

        # normalize to 0 and 1
        img = normalize(img)
    else:
        img = np.zeros((height,width))

    return img

def heatmap_generator(file_name, SCALE = 32):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    point_outs = []

    for point in points:
        point_out = [point["x"],point["y"]]
        point_outs.append(point_out)

    img = points_to_gaussian_heatmap(point_outs, height, width, SCALE)
    return img

class Medical_Data_test(Dataset):
    def __init__(self, data_path, data_mode, set_mode="test"):
        '''
        data_path: data path.
        data_mode: simulator or intra data.
        set_mode:  train or valid or test.
        transform: for data augmentation
        '''
        self.data_path = data_path
        self.data_mode = data_mode
        self.set_mode = set_mode
        self.imgs_path = glob.glob(os.path.join(data_path,"*/*/*.png"))

        print('Finished reading the {}_{} set of medical dataset ({} samples found)'
            .format(data_mode, set_mode, len(self.imgs_path)))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]

        image = Image.open(image_path).convert("RGB")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.imgs_path)

if __name__ == "__main__":
    simulator_dataset = Medical_Data_test("./input/","intra","test")
    simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    dataiter = iter(simulator_loader)
    images,  image_path= dataiter.next()
    print(image_path)
    print(images.shape)
