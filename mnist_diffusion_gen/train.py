# This code is modified from https://github.com/haldersourav/mnist-diffusion

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
from network import DDPM, ContextUnet


if __name__ == "__main__":
    n_epoch = 100
    batch_size = 64
    n_T = 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 256
    lrate = 2e-4
    save_model = True
    save_dir = './model/'
    ws_test = [0.0, 0.5, 2.0]

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()])

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=n_epoch,
        eta_min=1e-6
    )

    LOSS = []
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            LOSS.append(loss_ema)
            optim.step()

        with torch.no_grad():
            # save model
            if ep%10 == 0 or ep==n_epoch-1:
                torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
                print('saved model at ' + save_dir + f"model_{ep}.pth")
            ddpm.eval()
            temp_dataloader = DataLoader(dataset, batch_size=36, shuffle=False, num_workers=5)
            temp_x, temp_c = next(iter(temp_dataloader))
            x_gen, x_gen_store = ddpm.prediction(36, (1, 28, 28), temp_c, device, guide_w=2.0)
            save_image(x_gen, './' + f"prediction_{ep}.png", nrow=int(6))

        scheduler.step()
    np.save("./loss.npy", LOSS)



