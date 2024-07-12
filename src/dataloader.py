import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Optional


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: str, num_imgs: Optional[int] = None):
        self.img_dir = img_dir
        
        self.images = []
        img_files = [x for x in os.listdir(self.img_dir) if x.endswith(".jpg")]
        if num_imgs:
            num_imgs = min(num_imgs, len(img_files))
        else:
            num_imgs = len(img_files)

        img_files = img_files[:num_imgs] 

        for file in img_files:
            full_path = os.path.join(self.img_dir, file)
            self.images.append(full_path)
        
        self.num_imgs = num_imgs
        
    def __len__(self):
        return self.num_imgs
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path) / 127.5 - 1
        return image
        