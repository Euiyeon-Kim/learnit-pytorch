import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from model import SirenModel

NUM_ITERS = 1000


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


if __name__ == '__main__':
    os.makedirs('tmp', exist_ok=True)
    img = Image.open('cat/flickr_cat_000040.jpg')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)
    in_f = grid.shape[-1]

    model = SirenModel(coord_dim=in_f, num_c=c).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for i in range(NUM_ITERS):
        model.train()
        optim.zero_grad()

        pred = model(grid)
        loss = .5 * loss_fn(pred, img)

        loss.backward()
        optim.step()

        if (i+1) % 10 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'tmp/{i}_{loss.item()}.jpg')




