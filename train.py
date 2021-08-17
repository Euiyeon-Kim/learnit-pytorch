import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import MAML
from dataset import Custom

EXP_NAME = 'a'
DATA_ROOT = './A'

RES = 256
BATCH_SIZE = 1
INNER_STEPS = 2
MAX_ITERS = 150000

OUTER_LR = 1e-5
INNER_LR = 1e-2


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)

    dataset = Custom(DATA_ROOT, RES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(RES, RES, device=device)
    in_f = grid.shape[-1]

    maml = MAML(coord_dim=in_f, num_c=1).to(device)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = torch.nn.MSELoss()

    # Outer loops
    outer_step = 0
    while outer_step < MAX_ITERS:
        for data in dataloader:
            if outer_step > MAX_ITERS:
                break

            data = data.to(device)

            maml.train()
            pred = maml(grid, data, True)

            loss = loss_fn(pred, data)
            print(f'{outer_step}/{MAX_ITERS}: {loss.item()}')

            outer_optimizer.zero_grad()
            loss.backward()
            outer_optimizer.step()

            outer_step += 1

            if outer_step % 10 == 0:
                pred = pred[0].permute(2, 0, 1)
                data = data[0].permute(2, 0, 1)
                save_image(pred, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item()}_pred.jpg')
                save_image(data, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item()}_data.jpg')

            if outer_step % 100 == 0:
                torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/{outer_step}.pth')
