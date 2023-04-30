import yaml
import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from models import UNet
from torch.optim.lr_scheduler import _LRScheduler
from argparse import ArgumentParser
from fastmrt.data.dataset import SliceDataset
from fastmrt.utils.fftc import ifft2c_numpy
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn


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
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images in tqdmDataLoader:
                # train
                optimizer.zero_grad()                                    # 清空过往梯度
                x_0 = images.to(device)                                  # 将输入图像加载到计算设备上
                loss = trainer(x_0).sum() / 1000.                        # 前向传播并计算损失
                loss.backward()                                          # 反向计算梯度
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config["grad_clip"])    # 裁剪梯度，防止梯度爆炸
                optimizer.step()                                         # 更新参数
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


def eval(config: Dict):

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
        
        # sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[config["batch_size"], 3, 32, 32], device=device)
        sampledImgs = sampler(noisyImage)

        sampledImgs = sampledImgs * config["mean"][config["subset"]] + config["std"][config["subset"]]
        
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
    config["data_dir"] = os.path.join(config["data_dir"], args.subset, "train")
    config["save_dir"] = os.path.join(config["save_dir"], args.subset)
    if os.path.exists(config["save_dir"]) is False:
        os.mkdir(config["save_dir"])

    # launch train or eval process
    if args.stage == "train":
        train(config)
    else:
        eval(config)

