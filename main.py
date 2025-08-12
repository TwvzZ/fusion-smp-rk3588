import os, csv, random, argparse
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(img: Image.Image, size: Tuple[int,int]):
    # 转灰度 & resize & [0,1] 归一化 -> [1,H,W] tensor
    img = img.convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

# ---------------- SSIM & Sobel（缓存卷积核，避免反复创建） ----------------
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        kx = self.kx.to(dtype=x.dtype, device=x.device)
        ky = self.ky.to(dtype=x.dtype, device=x.device)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-6)

class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size//2
        g = torch.exp(-(coords**2)/(2*sigma*sigma))
        g = (g / g.sum()).unsqueeze(0)
        kernel = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # [1,1,11,11]
        self.register_buffer('window', kernel)  # 单通道
        self.C1 = C1; self.C2 = C2

    def forward(self, img1, img2):
        # img*: [N,1,H,W] in [0,1]
        window = self.window.to(dtype=img1.dtype, device=img1.device)
        mu1 = F.conv2d(img1, window, padding=window.size(-1)//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window.size(-1)//2, groups=1)
        mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window.size(-1)//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window.size(-1)//2, groups=1) - mu2_sq
        sigma12   = F.conv2d(img1*img2, window, padding=window.size(-1)//2, groups=1) - mu12

        C1, C2 = self.C1, self.C2
        ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

# ---------------- Dataset ----------------
class PairListDataset(Dataset):
    def __init__(self, csv_file: str, size=(256,256)):
        self.size = size
        self.items = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row['ir_path'], row['vis_path']))
        assert len(self.items)>0, "Empty CSV pairs!"
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ir_p, vis_p = self.items[idx]
        ir  = Image.open(ir_p)
        vis = Image.open(vis_p)
        ir_t  = to_tensor(ir,  self.size)   # [1,H,W]
        vis_t = to_tensor(vis, self.size)   # [1,H,W]
        x = torch.cat([ir_t, vis_t], dim=0) # [2,H,W]
        return x, ir_t, vis_t

# ---------------- Loss ----------------
class FusionLoss(nn.Module):
    def __init__(self, w_l1=0.5, w_ssim=0.5, w_grad=1.0):
        super().__init__()
        self.w_l1 = w_l1; self.w_ssim = w_ssim; self.w_grad = w_grad
        self.l1 = nn.L1Loss()
        self.ssim = SSIM()
        self.sobel = SobelGrad()
    def forward(self, fused, ir, vis):
        # L1 到两模态
        l1_loss = self.l1(fused, ir) + self.l1(fused, vis)
        # SSIM 越大越好 -> 1-ssim
        ssim_loss = (1.0 - self.ssim(fused, ir)) + (1.0 - self.ssim(fused, vis))
        # 边缘保持：让 fused 的梯度接近 max(∇IR, ∇VIS)
        g_f = self.sobel(fused); g_ir = self.sobel(ir); g_vis = self.sobel(vis)
        g_target = torch.maximum(g_ir, g_vis)
        grad_loss = self.l1(g_f, g_target)
        return self.w_l1*l1_loss + self.w_ssim*ssim_loss + self.w_grad*grad_loss

# ---------------- Train/Eval ----------------
@dataclass
class CFG:
    train_csv: str = "data/train_pairs.csv"
    val_csv:   str = "data/val_pairs.csv"
    img_size: Tuple[int,int] = (256,256)
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    num_workers: int = 4
    encoder_name: str = "mobilenet_v3_small"
    encoder_weights: str = None   # ⚠️ 与 in_channels=2 配套，禁用预训练
    in_channels: int = 2
    out_channels: int = 1
    amp: bool = True
    save_dir: str = "checkpoints"
    onnx_path: str = "fusion_unet_mbv3.onnx"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def build_model(cfg: CFG):
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,  # None
        in_channels=cfg.in_channels,          # 2 通道（IR+VIS）
        classes=cfg.out_channels,             # 1 通道输出
        encoder_depth=4,
        decoder_channels=[128, 64, 32, 16],
        activation='sigmoid',                 # ✅ 输出归一化到[0,1]
    )
    return model

def train_one_epoch(model, loader, optim, scaler, loss_fn, device):
    model.train()
    running = 0.0
    for x, ir, vis in tqdm(loader, desc="train", leave=False):
        x, ir, vis = x.to(device), ir.to(device), vis.to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            fused = model(x)         # 已 sigmoid
            loss = loss_fn(fused, ir, vis)
        optim.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward(); optim.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    ssim_ir_total, ssim_vis_total = 0.0, 0.0
    loss_total, n = 0.0, 0
    for x, ir, vis in tqdm(loader, desc="val", leave=False):
        x, ir, vis = x.to(device), ir.to(device), vis.to(device)
        fused = model(x)
        # 复用 loss 里的 ssim/sobel，算个参考
        loss = loss_fn(fused, ir, vis).item()
        ssim_ir  = loss_fn.ssim(fused, ir).item()
        ssim_vis = loss_fn.ssim(fused, vis).item()
        bsz = x.size(0)
        loss_total     += loss    * bsz
        ssim_ir_total  += ssim_ir * bsz
        ssim_vis_total += ssim_vis* bsz
        n += bsz
    return loss_total/n, ssim_ir_total/n, ssim_vis_total/n

def export_onnx(model, cfg: CFG):
    model.eval()
    dummy = torch.randn(1, cfg.in_channels, *cfg.img_size, device=cfg.device)
    torch.onnx.export(
        model, dummy, cfg.onnx_path,
        input_names=["input"], output_names=["fused"],
        opset_version=11, dynamic_axes=None  # 静态尺寸，便于 RKNN
    )
    print(f"[OK] Exported ONNX to: {cfg.onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=CFG.train_csv)
    parser.add_argument("--val_csv", default=CFG.val_csv)
    parser.add_argument("--img_size", type=int, nargs=2, default=CFG.img_size)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--encoder", default=CFG.encoder_name)
    parser.add_argument("--weights", default=None)  # 默认 None
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--onnx_path", default=CFG.onnx_path)
    args = parser.parse_args()

    cfg = CFG(train_csv=args.train_csv, val_csv=args.val_csv,
              img_size=tuple(args.img_size), epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              encoder_name=args.encoder, encoder_weights=args.weights or None,
              onnx_path=args.onnx_path)

    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    train_ds = PairListDataset(cfg.train_csv, cfg.img_size)
    val_ds   = PairListDataset(cfg.val_csv,   cfg.img_size)
    pin_mem = (torch.device(cfg.device).type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_mem)

    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    loss_fn = FusionLoss(w_l1=0.5, w_ssim=0.5, w_grad=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type=="cuda") else None

    best_score = -1e9
    for epoch in range(cfg.epochs):
        tr_loss = train_one_epoch(model, train_dl, optim, scaler, loss_fn, device)
        val_loss, ssim_ir, ssim_vis = evaluate(model, val_dl, loss_fn, device)
        score = (ssim_ir + ssim_vis) / 2.0
        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | SSIM(IR)={ssim_ir:.4f} SSIM(VIS)={ssim_vis:.4f}")
        # 保存最优
        if score > best_score:
            best_score = score
            path = os.path.join(cfg.save_dir, "best.pth")
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, path)
            print(f"[BEST] saved to {path} (score={best_score:.4f})")

    # 导出 ONNX（可选）
    if args.export_onnx:
        export_onnx(model, cfg)

if __name__ == "__main__":
    main()
