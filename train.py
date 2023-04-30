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
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
    if config["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            config["save_weight_dir"], config["training_load_weight"]), map_location=device))
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
            for images, labels in tqdmDataLoader:
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
        warmUpScheduler.step()                                           # 调度器更新学习率
        torch.save(net_model.state_dict(), os.path.join(
            config["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))  # 保存模型


def eval(config: Dict):
    # load model and evaluate
    with torch.no_grad():
        # 建立和加载模型
        device = torch.device(config["device"])
        model = UNet(T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"], attn=config["attn"],
                     num_res_blocks=config["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            config["save_weight_dir"], config["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        # 实例化反向扩散采样器
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, config["beta_1"], config["beta_T"], config["T"]).to(device)
        # Sampled from standard normal distribution
        # 随机生成高斯噪声图像并保存
        noisyImage = torch.randn(
            size=[config["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            config["sampled_dir"], config["sampledNoisyImgName"]), nrow=config["nrow"])
        # 反向扩散并保存输出图像
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            config["sampled_dir"],  config["sampledImgName"]), nrow=config["nrow"])
        
if __name__ == '__main__':
    
    with open('./config.yaml') as fconfig:
        config = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    if config.pop("state") == "train":
        train(config)
    else:
        eval(config)
    train(config=config)

