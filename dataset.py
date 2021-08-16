from glob import glob

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class Custom(Dataset):
    def __init__(self, data_root, res):
        self.res = res
        self.paths = glob(f'{data_root}/*.jpg')
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = img.resize((self.res, self.res), Image.BICUBIC)
        img = np.float32(img) / 255
        return img


if __name__ == '__main__':
    dataset = Custom('./cat', 64)
    from torch.utils.data import DataLoader
    cat_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    print(len(dataset), len(cat_dataloader))
