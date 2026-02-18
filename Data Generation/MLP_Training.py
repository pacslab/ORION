# -*- coding: utf-8 -*-
import os, re, zipfile, urllib.request, random, argparse, time, platform, math, warnings, multiprocessing, io
from typing import Tuple, Callable, Dict, Any, List
from contextlib import nullcontext

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root  # type: ignore
    from torch.amp import GradScaler as _GradScaler   # type: ignore
    _USE_TORCH_AMP_ROOT = True

def _amp_autocast(enabled: bool):
    if not enabled: return nullcontext()
    if _USE_TORCH_AMP_ROOT:
        return _autocast_root(device_type='cuda', dtype=torch.float16)
    else:
        return _autocast_cuda(dtype=torch.float16)

FIXED = {
    "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.05},
    "sgd": {"momentum": 0.9, "nesterov": False, "weight_decay": 5e-4},
    "dropout_p": 0.0,
    "activation": "gelu",
}

def available_cpu_ids() -> List[int]:
    try:
        return sorted(list(os.sched_getaffinity(0)))
    except Exception:
        return list(range(multiprocessing.cpu_count()))

def apply_cpu_caps(k: int):
    os.environ["OMP_NUM_THREADS"] = str(k)
    os.environ["MKL_NUM_THREADS"] = str(k)
    os.environ["OPENBLAS_NUM_THREADS"] = str(k)
    os.environ["NUMEXPR_NUM_THREADS"] = str(k)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        torch.set_num_threads(k)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        ids = available_cpu_ids()
        os.sched_setaffinity(0, set(ids[:k]))
    except Exception:
        pass

_TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
_TINY_ZIP = "tiny-imagenet-200.zip"
_TINY_DIR = "tiny-imagenet-200"

def _download_tiny_imagenet(root: str) -> str:
    root = os.path.abspath(root); os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, _TINY_DIR)
    if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, "train")):
        return target_dir
    zip_path = os.path.join(root, _TINY_ZIP)
    if not os.path.exists(zip_path):
        print("Downloading TinyImageNet-200 (~236MB)...")
        urllib.request.urlretrieve(_TINY_URL, zip_path)
    print("Extracting TinyImageNet-200...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return target_dir

def _read_bytes_uncached(path: str, chunk_bytes: int = 262144) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except Exception: pass
        out = bytearray()
        while True:
            b = os.read(fd, chunk_bytes)
            if not b: break
            out += b
        try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception: pass
        return bytes(out)
    finally:
        os.close(fd)

class TinyImageNet200(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Callable = None, ssd_mode: bool = False):
        assert split in ("train", "val")
        self.root = _download_tiny_imagenet(root); self.split = split; self.transform = transform; self.ssd_mode = ssd_mode
        with open(os.path.join(self.root, "wnids.txt"), "r") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            train_dir = os.path.join(self.root, "train")
            for wnid in self.wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir): continue
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        else:
            val_dir = os.path.join(self.root, "val")
            ann_path = os.path.join(val_dir, "val_annotations.txt"); mapping = {}
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2: mapping[parts[0]] = parts[1]
            img_dir = os.path.join(val_dir, "images")
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    wnid = mapping.get(fname, None)
                    if wnid is not None:
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        self.num_classes = 200
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        if self.ssd_mode:
            raw = _read_bytes_uncached(path, 262144)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, target

def tmf_fix(tfm): return tfm

def build_cifar100(data_root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
    ])
    ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tmf_fix(tfm))
    return ds, 100, (3, 32, 32), 'CIFAR100', len(ds)

def build_tiny_imagenet_train(data_root, ssd_mode: bool):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),(0.2770, 0.2691, 0.2821))
    ])
    ds = TinyImageNet200(root=data_root, split="train", transform=tmf_fix(tfm), ssd_mode=ssd_mode)
    return ds, 200, (3, 64, 64), 'TinyImageNet', len(ds)

def build_stl10_train(data_root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4409, 0.4279, 0.3868),(0.2683, 0.2610, 0.2687))
    ])
    ds = datasets.STL10(root=data_root, split='train', download=True, transform=tmf_fix(tfm))
    return ds, 10, (3, 96, 96), 'STL10', len(ds)

DATASET_BUILDERS = {
    'CIFAR100': build_cifar100,
    'TinyImageNet': build_tiny_imagenet_train,
    'STL10': build_stl10_train,
}
ALLOWED_DATASETS_BY_MODEL = {
    'mixer':   ['TinyImageNet', 'STL10'],
    'resmlp':  ['CIFAR100', 'TinyImageNet'],
    'asmlp':   ['TinyImageNet', 'STL10'],
}

class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = in_chans * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(self.patch_dim, embed_dim)
    def forward(self, x):
        x = self.unfold(x).transpose(1, 2)
        return self.proj(x)

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_mlp_dim, channel_mlp_dim, p=0.0, act='gelu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU() if act == 'gelu' else nn.ReLU(), nn.Dropout(p),
            nn.Linear(token_mlp_dim, num_patches), nn.Dropout(p),
        )
        self.ln2 = nn.LayerNorm(dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim), the_act, nn.Dropout(p),
            nn.Linear(channel_mlp_dim, dim), nn.Dropout(p),
        )
    def forward(self, x):
        y = self.token_mlp(self.ln1(x).transpose(1, 2)).transpose(1, 2); x = x + y
        y = self.channel_mlp(self.ln2(x)); return x + y

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=8, token_mlp_dim=128, channel_mlp_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        P = self.patch.num_patches
        self.blocks = nn.Sequential(*[MixerBlock(P, embed_dim, token_mlp_dim, channel_mlp_dim, p=p, act=act) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim); self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.patch(x); x = self.blocks(x); x = self.ln(x).mean(dim=1); return self.head(x)

class ResMLPBlock(nn.Module):
    def __init__(self, num_patches, dim, hidden_channel_dim, p=0.0, act='gelu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_linear = nn.Linear(num_patches, num_patches)
        self.drop1 = nn.Dropout(p); self.ln2 = nn.LayerNorm(dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_channel_dim), the_act, nn.Dropout(p),
            nn.Linear(hidden_channel_dim, dim), nn.Dropout(p),
        )
    def forward(self, x):
        y = self.token_linear(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.drop1(y); y = self.channel_mlp(self.ln2(x)); return x + y

class ResMLP(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=12, hidden_channel_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        P = self.patch.num_patches
        self.blocks = nn.Sequential(*[ResMLPBlock(P, embed_dim, hidden_channel_dim, p=p, act=act) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim); self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.patch(x); x = self.blocks(x); x = self.ln(x).mean(dim=1); return self.head(x)

class ASMLPBlock(nn.Module):
    def __init__(self, grid_size: int, dim: int, hidden_channel_dim: int, p=0.0, act='gelu'):
        super().__init__()
        self.grid = grid_size; self.dim = dim; assert dim % 4 == 0
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p); self.proj = nn.Linear(dim * 2, dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_channel_dim), the_act, nn.Dropout(p),
            nn.Linear(hidden_channel_dim, dim), nn.Dropout(p),
        )
    def forward(self, x):
        N, P, D = x.shape; H = W = self.grid
        y = self.ln1(x).view(N, H, W, D); Cg = D // 4
        y1, y2, y3, y4 = y[..., :Cg], y[..., Cg:2*Cg], y[..., 2*Cg:3*Cg], y[..., 3*Cg:]
        y1 = torch.roll(y1, shifts=1, dims=1); y2 = torch.roll(y2, shifts=-1, dims=1)
        y3 = torch.roll(y3, shifts=1, dims=2); y4 = torch.roll(y4, shifts=-1, dims=2)
        y_shift = torch.cat([y1, y2, y3, y4], dim=-1).view(N, P, D)
        y = torch.cat([y_shift, x], dim=-1); y = self.proj(y)
        x = x + self.drop(y); y = self.channel_mlp(self.ln2(x)); return x + y

class ASMLP(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=12, hidden_channel_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        grid = int(math.sqrt(self.patch.num_patches))
        self.blocks = nn.Sequential(*[ASMLPBlock(grid, embed_dim, hidden_channel_dim, p=p, act=act) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim); self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.patch(x); x = self.blocks(x); x = self.ln(x).mean(dim=1); return self.head(x)

def build_image_mlp(model_name: str, image_size: int, in_chans: int, num_classes: int,
                    patch_size: int, embed_dim: int, depth: int,
                    token_mlp_dim: int, channel_mlp_dim: int) -> nn.Module:
    name = model_name.lower(); p = FIXED["dropout_p"]; act = FIXED["activation"]
    if name == 'mixer':
        return MLPMixer(image_size, patch_size, in_chans, num_classes, embed_dim, depth,
                        token_mlp_dim, channel_mlp_dim, p, act)
    elif name == 'resmlp':
        return ResMLP(image_size, patch_size, in_chans, num_classes, embed_dim, depth,
                      channel_mlp_dim, p, act)
    elif name == 'asmlp':
        if embed_dim % 4 != 0: raise ValueError("embed_dim must be divisible by 4.")
        return ASMLP(image_size, patch_size, in_chans, num_classes, embed_dim, depth,
                     channel_mlp_dim, p, act)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def _safe_path_component(s: str, maxlen: int = 140) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', str(s)); s = re.sub(r'\s+', '_', s.strip()); return s[:maxlen]

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gpu_info():
    if DEVICE.type == 'cuda':
        prop = torch.cuda.get_device_properties(0); return prop.name
    return 'CPU'

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

def build_loader(ds, batch_size: int, num_workers: int = 2, ssd_mode: bool = False) -> DataLoader:
    if platform.system() == "Windows":
        num_workers = 0
    kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if DEVICE.type == 'cuda':
        kwargs["pin_memory"] = (not ssd_mode)
    if num_workers > 0:
        return DataLoader(persistent_workers=False, prefetch_factor=(1 if ssd_mode else 2), **kwargs)
    else:
        return DataLoader(persistent_workers=False, **kwargs)

def estimate_flops_forward_per_sample(model_name: str, image_size: int, patch_size: int,
                                      in_chans: int, embed_dim: int, depth: int,
                                      token_mlp_dim: int, channel_mlp_dim: int) -> Dict[str, Any]:
    P = (image_size // patch_size) ** 2; D = embed_dim; patch_dim = in_chans * patch_size * patch_size
    flops_patch = P * (2.0 * patch_dim * D); flops_ln_block = 4.0 * P * D
    name = model_name.lower()
    if name == 'mixer':
        flops_token = 4.0 * D * P * token_mlp_dim; flops_channel = 4.0 * P * D * channel_mlp_dim
        flops_block = flops_token + flops_channel + flops_ln_block
    elif name == 'resmlp':
        flops_token = 2.0 * D * P * P; flops_channel = 4.0 * P * D * channel_mlp_dim
        flops_block = flops_token + flops_channel + flops_ln_block
    elif name == 'asmlp':
        flops_proj = P * (4.0 * D * D); flops_channel = 4.0 * P * D * channel_mlp_dim
        flops_block = flops_proj + flops_channel + flops_ln_block
    else:
        raise ValueError(f"Unknown model for FLOPs: {model_name}")
    flops_fwd = flops_patch + depth * flops_block
    return {'num_patches': int(P), 'flops_fwd_per_sample': float(flops_fwd)}

def estimate_memory_bytes(param_cnt: int, precision_bits: int, optimizer_name: str, amp: bool) -> Dict[str, Any]:
    param_bytes = param_cnt * 4
    opt = optimizer_name.lower(); opt_mult = 2 if opt in ['adam', 'adamw'] else 1
    optimizer_state_bytes = param_bytes * opt_mult
    master_param_bytes = param_bytes if amp else 0
    total_state_bytes = param_bytes + optimizer_state_bytes + master_param_bytes
    act_bytes_per_scalar = 2 if precision_bits == 16 else 4
    return {
        'param_bytes': int(param_bytes),
        'optimizer_state_bytes': int(optimizer_state_bytes),
        'master_param_bytes': int(master_param_bytes),
        'total_state_bytes': int(total_state_bytes),
        'act_bytes_per_scalar': int(act_bytes_per_scalar),
    }

def _build_optimizer(params, name: str, lr: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=FIXED["adamw"]["betas"],
                                 eps=FIXED["adamw"]["eps"], weight_decay=FIXED["adamw"]["weight_decay"])
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=FIXED["sgd"]["momentum"],
                               weight_decay=FIXED["sgd"]["weight_decay"], nesterov=FIXED["sgd"]["nesterov"])
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def measure_batch_timing(model: nn.Module, train_loader: DataLoader,
                         warmup_batches: int = 5, measure_batches: int = 20,
                         lr: float = 1e-3, optimizer_name: str = 'AdamW',
                         amp: bool = False) -> Dict[str, Any]:
    model.to(DEVICE)
    opt = _build_optimizer(model.parameters(), optimizer_name, lr)
    scaler = _GradScaler(enabled=(amp and DEVICE.type == 'cuda'))
    total_step_ms, compute_ms_list, loader_ms_list = [], [], []
    it = iter(train_loader); total_batches = warmup_batches + measure_batches
    for i in range(total_batches):
        t0_load = time.perf_counter()
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader); xb, yb = next(it)
        xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True).long()
        t1_load = time.perf_counter(); loader_ms = (t1_load - t0_load) * 1000.0
        opt.zero_grad(set_to_none=True)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t0_compute = time.perf_counter()
        if scaler.is_enabled():
            with _amp_autocast(True):
                logits = model(xb); loss = nn.functional.cross_entropy(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            logits = model(xb); loss = nn.functional.cross_entropy(logits, yb); loss.backward(); opt.step()
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t1_compute = time.perf_counter()
        compute_ms = (t1_compute - t0_compute) * 1000.0
        total_ms = loader_ms + compute_ms
        if i >= warmup_batches:
            total_step_ms.append(total_ms); loader_ms_list.append(loader_ms); compute_ms_list.append(compute_ms)
    return dict(
        avg_batch_ms=float(np.mean(total_step_ms)) if total_step_ms else float('nan'),
        p90_batch_ms=float(np.percentile(total_step_ms, 90)) if total_step_ms else float('nan'),
        avg_loader_ms=float(np.mean(loader_ms_list)) if loader_ms_list else float('nan'),
        avg_compute_ms=float(np.mean(compute_ms_list)) if compute_ms_list else float('nan'),
    )

def main():
    p = argparse.ArgumentParser("Image MLP Timing Grid")
    p.add_argument('--datasets', nargs='+', default=['CIFAR100', 'TinyImageNet', 'STL10'])
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--logdir', type=str, default='./logs')
    p.add_argument('--warmup-batches', type=int, default=5)
    p.add_argument('--measure-batches', type=int, default=20)
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[32, 128, 256])
    p.add_argument('--lrs', nargs='+', type=float, default=[5e-4, 1e-3, 2e-3])
    p.add_argument('--optimizers', nargs='+', default=['AdamW','SGD'])
    p.add_argument('--precisions', nargs='+', type=int, default=[16, 32], choices=[16, 32])
    p.add_argument('--worker-sweep', nargs='+', type=int, default=[4, 8, 16])
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--vcpu-cap', type=int, default=None)
    p.add_argument('--storage', type=str, choices=['SSD', 'NVME'], default=None)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--models', nargs='+', default=['mixer', 'resmlp', 'asmlp'])
    p.add_argument('--embed-dims', nargs='+', type=int, default=[128, 256])
    p.add_argument('--depth', type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    abs_logdir = os.path.abspath(args.logdir)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(args.seed)

    host_cpu_ids = available_cpu_ids()
    cpu_cap = args.vcpu_cap if args.vcpu_cap is not None else len(host_cpu_ids)
    cpu_cap = max(1, min(cpu_cap, len(host_cpu_ids)))
    apply_cpu_caps(cpu_cap)

    ds_names = [name for name in args.datasets if name in DATASET_BUILDERS]
    gpu_name_str = gpu_info()
    out_name = f"MLP_{_safe_path_component(gpu_name_str)}.csv" if DEVICE.type == 'cuda' else "MLP_CPU.csv"
    rows = []

    worker_values = list(dict.fromkeys(args.worker_sweep)) if args.worker_sweep else [args.num_workers or 2]
    if args.num_workers is not None:
        worker_values = [int(args.num_workers)]
    worker_values = [max(0, min(int(w), cpu_cap)) for w in worker_values]

    storage_label = 'NVME' if (args.storage is None or args.storage.upper() == 'NVME') else 'SSD'
    ssd_mode = (storage_label == 'SSD')

    for model_name in args.models:
        allowed_for_model = set(ALLOWED_DATASETS_BY_MODEL.get(model_name.lower(), []))
        model_ds_list = [d for d in ds_names if d in allowed_for_model]
        if not model_ds_list:
            print(f"[WARN] No allowed datasets selected for model '{model_name}'. Skipping."); continue

        for ds_name in model_ds_list:
            if ds_name == 'TinyImageNet':
                ds, num_classes, input_shape, ds_label, total_N = build_tiny_imagenet_train(args.data_root, ssd_mode=ssd_mode)
            elif ds_name == 'CIFAR100':
                ds, num_classes, input_shape, ds_label, total_N = build_cifar100(args.data_root)
            else:
                ds, num_classes, input_shape, ds_label, total_N = build_stl10_train(args.data_root)

            in_chans, H, W = input_shape
            if ds_name == "CIFAR100":
                image_size = 32; patch_size = 4 if model_name == 'asmlp' else 8
                default_depth_for = lambda m: (8 if m == 'mixer' else 12)
            elif ds_name == "TinyImageNet":
                image_size = 64; patch_size = 8
                default_depth_for = lambda m: (10 if m == 'mixer' else 12)
            elif ds_name == "STL10":
                image_size = 96; patch_size = 8
                default_depth_for = lambda m: (8 if m == 'mixer' else 12)
            else:
                raise ValueError("Unsupported dataset")

            trials: List[Dict[str, Any]] = []
            for embed_dim in args.embed_dims:
                channel_mlp_dim = 4 * embed_dim
                token_mlp_dim = max(128, embed_dim // 2)
                depth = args.depth if args.depth is not None else default_depth_for(model_name)
                for lr in args.lrs:
                    for B in args.batch_sizes:
                        for optimizer_name in args.optimizers:
                            for precision in args.precisions:
                                for workers in worker_values:
                                    trials.append({
                                        'embed_dim': embed_dim,
                                        'channel_mlp_dim': channel_mlp_dim,
                                        'token_mlp_dim': token_mlp_dim,
                                        'depth': depth,
                                        'lr': lr, 'batch_size': B,
                                        'optimizer_name': optimizer_name,
                                        'precision': precision,
                                        'num_workers': int(workers),
                                    })

            for t in tqdm(trials, desc=f"{ds_name} — {model_name}", unit="trial"):
                embed_dim = t['embed_dim']; channel_mlp_dim = t['channel_mlp_dim']
                token_mlp_dim = t['token_mlp_dim']; depth = t['depth']
                lr = t['lr']; B = t['batch_size']; optimizer_name = t['optimizer_name']
                precision = t['precision']; num_workers = t['num_workers']

                flops_info = estimate_flops_forward_per_sample(
                    model_name=model_name, image_size=image_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=embed_dim, depth=depth, token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim
                )
                P_patches = flops_info['num_patches']
                flops_fwd_per_sample = flops_info['flops_fwd_per_sample']
                flops_train_per_sample = flops_fwd_per_sample * 3.0

                try:
                    loader = build_loader(ds, batch_size=B, num_workers=num_workers, ssd_mode=ssd_mode)
                except RuntimeError as e:
                    print(f"[ERROR] DataLoader failed for B={B}, workers={num_workers}: {e}"); continue

                try:
                    model = build_image_mlp(model_name, image_size, in_chans, num_classes, patch_size,
                                            embed_dim, depth, token_mlp_dim, channel_mlp_dim)
                except Exception as e:
                    print(f"[ERROR] Model build failed ({model_name}) on {ds_name}: {e}"); continue

                P_params = param_count(model)
                use_amp = (precision == 16) and (DEVICE.type == 'cuda')

                mem_info = estimate_memory_bytes(P_params, precision, optimizer_name, use_amp)
                act_bytes_per_sample_proxy = P_patches * embed_dim * mem_info['act_bytes_per_scalar'] * depth
                flops_train_per_batch = flops_train_per_sample * B

                try:
                    out = measure_batch_timing(model, loader,
                                               warmup_batches=args.warmup_batches,
                                               measure_batches=args.measure_batches,
                                               lr=lr, optimizer_name=optimizer_name, amp=use_amp)
                except RuntimeError as e:
                    print(f"[ERROR] Training failed on {ds_name} [{model_name}] (B={B}, lr={lr}, opt={optimizer_name}, workers={num_workers}): {e}")
                    del model
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                    continue

                T_gpu_ms = float(out['avg_compute_ms'])
                T_cpu_io_ms = float(out['avg_loader_ms'])
                T_step_ms = float(max(T_gpu_ms, T_cpu_io_ms))
                loader_ratio = float(T_cpu_io_ms / (T_cpu_io_ms + T_gpu_ms)) if (T_cpu_io_ms + T_gpu_ms) > 0 else float('nan')
                bottleneck = 'gpu' if T_gpu_ms > T_cpu_io_ms else 'host'

                rows.append({
                    'dataset': ds_label,
                    'input_shape': str(input_shape),
                    'num_classes': int(num_classes),
                    'dataset_size': int(total_N),

                    'learning_rate': float(lr),
                    'batch_size': int(B),
                    'optimizer': optimizer_name,
                    'precision': int(precision),
                    'amp': bool(use_amp),

                    'model': model_name,
                    'architecture': 'MLP',
                    'embed_dim': int(embed_dim),
                    'depth': int(depth),
                    'patch_size': int(patch_size),
                    'token_mlp_dim': int(token_mlp_dim),
                    'channel_mlp_dim': int(channel_mlp_dim),
                    'num_patches': int(P_patches),
                    'param_count': int(P_params),

                    'avg_batch_time_ms': float(out['avg_batch_ms']),
                    'p90_batch_time_ms': float(out['p90_batch_ms']),
                    'T_gpu_ms': T_gpu_ms,
                    'T_cpu_io_ms': T_cpu_io_ms,
                    'T_step_ms': T_step_ms,
                    'loader_ratio': loader_ratio,
                    'bottleneck': bottleneck,

                    'flops_fwd_per_sample': float(flops_fwd_per_sample),
                    'flops_train_per_sample': float(flops_train_per_sample),
                    'flops_train_per_batch': float(flops_train_per_batch),
                    'param_bytes': int(mem_info['param_bytes']),
                    'optimizer_state_bytes': int(mem_info['optimizer_state_bytes']),
                    'master_param_bytes': int(mem_info['master_param_bytes']),
                    'total_state_bytes': int(mem_info['total_state_bytes']),
                    'act_bytes_per_sample_proxy': int(act_bytes_per_sample_proxy),
                    'arithmetic_intensity_train': float(
                        flops_train_per_sample / max(1.0, (mem_info['param_bytes'] + act_bytes_per_sample_proxy))
                    ),

                    'gpu_name': gpu_name_str if DEVICE.type == 'cuda' else 'CPU',
                    'vcpu': int(num_workers),
                    'storage': storage_label,
                })

                del model
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df = df.loc[:, ~df.columns.duplicated()]
    os.makedirs(abs_logdir, exist_ok=True)
    out_path = os.path.join(abs_logdir, out_name)
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved benchmark to:\n  {out_path}")

if __name__ == '__main__':
    main()
