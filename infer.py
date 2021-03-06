import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from train import create_grid
from model import SirenModel

EXP_NAME = 'cat'
PTH_NUM = 50000
RES = 178
PATH = './infer.jpg'
TEST_RANGE = 10
LR = 1e-2


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/maml/{PTH_NUM}', exist_ok=True)

    img = Image.open(PATH)
    img = img.resize((RES, RES), Image.BICUBIC)
    img = np.float32(img) / 255
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)
    in_f = grid.shape[-1]

    model = SirenModel(coord_dim=in_f, num_c=3).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NUM}.pth'))

    optim = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    pred = model(grid)
    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/maml/{PTH_NUM}/meta.jpg')

    for i in range(TEST_RANGE):
        model.train()
        optim.zero_grad()

        pred = model(grid)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        pred = pred.permute(2, 0, 1)
        save_image(pred, f'exps/{EXP_NAME}/maml/{PTH_NUM}/{i}_{loss.item():.4f}.jpg')