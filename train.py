import yaml
import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import h5py
import numpy as np

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from models import UNet
from torch.optim.lr_scheduler import _LRScheduler
from argparse import ArgumentParser
from fastmrt.data.dataset import SliceDataset
from fastmrt.utils.fftc import ifft2c_numpy, fft2c_numpy
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.trans import real_tensor_to_complex_np as rt2cn


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def train(config: Dict):
    device = torch.device(config["device"])
    # dataset
    dataset = SliceDataset(
        root=config["data_dir"],
        transform=transforms.Compose([
            lambda data: torch.from_numpy(cn2rn(ifft2c_numpy(data["kspace"]))),
            transforms.Normalize(torch.Tensor(config["mean"][config["subset"]]).unsqueeze(-1).unsqueeze(-1), 
                                 torch.Tensor(config["std"][config["subset"]]).unsqueeze(-1).unsqueeze(-1)),
            lambda data: torch.clip(data, min=-3., max=3.)
        ]))
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
    if config["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            config["save_dir"], config["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
    # 设置学习率衰减，按余弦函数的1/2个周期衰减，从``lr``衰减至0
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epoch"], eta_min=0, last_epoch=-1)
    # 设置逐步预热调度器，学习率从0逐渐增加至multiplier * lr，共用1/10总epoch数，后续学习率按``cosineScheduler``设置进行变化
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=config["multiplier"], warm_epoch=config["epoch"] // 10, after_scheduler=cosineScheduler)
    # 实例化训练模型
    trainer = GaussianDiffusionTrainer(
        net_model, config["beta_1"], config["beta_T"], config["T"]).to(device)

    # start training
    m=0.9
    loss = 1e3
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images in tqdmDataLoader:
                # train
                optimizer.zero_grad()                                    # 清空过往梯度
                x_0 = images.to(device)                                  # 将输入图像加载到计算设备上
                loss = m * loss + (1 - m) * (trainer(x_0).sum() / 1000.)
                loss.backward()                                          # 反向计算梯度
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config["grad_clip"])    # 裁剪梯度，防止梯度爆炸
                optimizer.step()                                         # 更新参数
                loss = loss.detach()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": f"{loss.item():.4g}",
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })                                                       # 设置进度条显示内容
        warmUpScheduler.step()
        if (e + 1) % config["save_interval"] == 0:
            torch.save(net_model.state_dict(), os.path.join(
                config["save_dir"], f"ckpt_{e+1}_.pt"))


def pred(config: Dict):

    with torch.no_grad():

        # build model and load checkpoint
        device = torch.device(config["device"])
        model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"],
                     num_res_blocks=config["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            config["save_dir"], config["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        print("model load weight done.")

        # build sampler
        sampler = GaussianDiffusionSampler(
            model, config["beta_1"], config["beta_T"], config["T"]).to(device)

        mean = torch.Tensor(config["mean"][config["subset"]]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)                                 
        std = torch.Tensor(config["std"][config["subset"]]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        for batch_idx in range(config["gen_num"] // config["batch_size"]):

            # sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[config["batch_size"], 2, 96, 96], device=device)
            sampled_imgs = sampler(noisyImage)

            # save datas
            sampled_imgs = sampled_imgs.cpu() * std + mean
            for idx in range(config["batch_size"]):

                # prepare saved data
                curr_idx = batch_idx * config["batch_size"] + idx
                curr_ksp = fft2c_numpy(rt2cn(sampled_imgs[idx]).transpose())[:, :, np.newaxis, np.newaxis, np.newaxis]  # to fastmrt-data-style: 5d-data
                header = dict(type="diffusion-augs", width=96, height=96, frames=1, slices=1, coils=1)
                
                # write
                with h5py.File(os.path.join(config["sampled_dir"], f"d{curr_idx:05d}.h5"), "w") as hf:
                    hf.create_dataset("kspace", data=curr_ksp, dtype="complex64")
                    hf.create_dataset("tmap_masks", data=None, dtype='<f4')
                    for key, value in header.items():
                        hf.attrs[key] = value
        
if __name__ == '__main__':

    # command line inplements
    parser = ArgumentParser()
    parser.add_argument("--stage", "-s", type=str, required=True, 
                        help="the stage you want to launch, one of `train` and `eval`.")
    parser.add_argument("--subset", "-ss", type=str, required=True,
                        help="the subset you want to train, one of `phantom`, `exvivo` and `invivo`.")
    parser.add_argument("--cfg", "-c", type=str, required=False, default='./config.yaml',
                        help="the directory of config file, default is `./config.yaml`.")
    args = parser.parse_args()
    
    # read config file
    with open(args.cfg) as fconfig:
        config = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    config["subset"] = args.subset
    config["data_dir"] = [os.path.join(d_dir, args.subset, "train") for d_dir in config["data_dir"]]
    config["save_dir"] = os.path.join(config["save_dir"], args.subset)
    config["sampled_dir"] = os.path.join(config["save_dir"], "datas")
    if os.path.exists(config["save_dir"]) is False:
        os.mkdir(config["save_dir"])
    if os.path.exists(config["sampled_dir"]) is False:
        os.mkdir(config["sampled_dir"])

    # launch train or eval process
    if args.stage == "train":
        train(config)
    elif args.stage == "pred":
        pred(config)

