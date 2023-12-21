
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        H, W = 512, 512
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 2)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        #image = image[::2,:,:]
        #image = image[1,:,:]
        #print(image.shape)
        #x = torch.zeros((3, 512, 512))
        #x[0,:,:] = image[1,:,:]
        #x[1,:,:] = image[1,:,:]
        #x[2,:,:] = image[1,:,:]
        #image = x

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        y = torch.zeros((3, 512, 512), dtype=torch.float32)
        y[0,:,:] = (mask==0) #(mask==70)+(mask==130) #background
        y[1,:,:] = (mask==70) #artery
        y[2,:,:] = (mask==130) #vein
        mask = y
        


        return image, mask

    def __len__(self):
        return self.n_samples
