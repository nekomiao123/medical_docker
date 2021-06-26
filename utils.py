import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    return image

def check_accuracy(loader, model, device="cuda"):
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, z in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))

            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    dice = dice_score/len(loader)
    model.train()
    return dice

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(loader, model, folder="pic/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()