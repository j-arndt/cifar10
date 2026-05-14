"""
hlb-cifar10 skeleton (tysam-code/hlb-cifar10) — MOAB baseline.
Agent's apply_fn(net, hyp) -> (net, hyp) to optimize the training run.
"""
import copy
import functools
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import yaml

torch._dynamo.config.suppress_errors = True  # fall back to eager on compile fail

# ── Default hyperparameters (agent can mutate these) ──────────────────────────
DEFAULT_HYP = {
    'opt': {
        'bias_lr':             1.525 * 64 / 512,
        'non_bias_lr':         1.525 / 512,
        'bias_decay':          6.687e-4 * 1024 / 64,
        'non_bias_decay':      6.687e-4 * 1024,
        'scaling_factor':      1. / 9,
        'percent_start':       0.23,
        'loss_scale_scaler':   1. / 32,
    },
    'net': {
        'whitening':           {'kernel_size': 2, 'num_examples': 50000},
        'batch_norm_momentum': 0.4,
        'cutmix_size':         3,
        'cutmix_epochs':       6,
        'pad_amount':          2,
        'base_depth':          64,
    },
    'misc': {
        'ema': {
            'epochs':          10,
            'decay_base':      0.95,
            'decay_pow':       3.0,
            'every_n_steps':   5,
        },
        'train_epochs':        12.1,
        'device':              'cuda',
        'data_location':       'data.pt',
    },
}

# ── Network components ─────────────────────────────────────────────────────────
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=0.4):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = False
        self.bias.requires_grad = True

class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**default_conv_kwargs, **kwargs})

class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    def forward(self, x):
        w = self.weight * self.temperature if self.temperature else self.weight
        return x @ w.T

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, bn_momentum=0.4):
        super().__init__()
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm1 = BatchNorm(channels_out, momentum=bn_momentum)
        self.norm2 = BatchNorm(channels_out, momentum=bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool1(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))

class FastGlobalMaxPooling(nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=(2, 3))

class SpeedyConvNet(nn.Module):
    def __init__(self, net_dict):
        super().__init__()
        self.net_dict = net_dict

    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['conv_group_1'](x)
        x = self.net_dict['conv_group_2'](x)
        x = self.net_dict['conv_group_3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        if not self.training:
            orig, flipped = x.split(x.shape[0] // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x

class NetworkEMA(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net_ema = copy.deepcopy(net).eval().requires_grad_(False)

    def update(self, current_net, decay):
        with torch.no_grad():
            for ema_p, (name, p) in zip(self.net_ema.state_dict().values(),
                                         current_net.state_dict().items()):
                if p.dtype in (torch.half, torch.float):
                    ema_p.mul_(decay).add_(p.detach() * (1.0 - decay))
                    if not ('norm' in name and 'weight' in name) and 'whiten' not in name:
                        p.copy_(ema_p.detach())

    def forward(self, x):
        with torch.no_grad():
            return self.net_ema(x)

# ── Whitening helpers ──────────────────────────────────────────────────────────
def get_patches(x, patch_shape=(3, 3)):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()

def get_whitening_parameters(patches):
    n, c, h, w = patches.shape
    cov = torch.cov(patches.view(n, c * h * w).t())
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c * h * w, c, h, w).flip(0)

def init_whitening_conv(layer, train_images, num_examples, pad_amount, whiten_splits=5000):
    if pad_amount > 0:
        data = train_images[:num_examples, :, pad_amount:-pad_amount, pad_amount:-pad_amount]
    else:
        data = train_images[:num_examples]
    eigenvalues, eigenvectors = [], []
    for chunk in data.split(whiten_splits, dim=0):
        ev, evec = get_whitening_parameters(get_patches(chunk, layer.weight.shape[2:]))
        eigenvalues.append(ev); eigenvectors.append(evec)
    ev  = torch.stack(eigenvalues).mean(0)
    evec = torch.stack(eigenvectors).mean(0)
    n = layer.weight.shape[0]
    sliced = (evec / torch.sqrt(ev + 1e-2))[-n:].to(layer.weight.dtype)
    layer.weight.data = torch.cat((sliced, -sliced), dim=0)
    layer.weight.requires_grad = False

# ── Data augmentation ──────────────────────────────────────────────────────────
def make_random_square_masks(inputs, mask_size):
    if mask_size == 0:
        return None
    is_even = int(mask_size % 2 == 0)
    s = inputs.shape
    cy = torch.empty(s[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, s[-2] - mask_size // 2 - is_even)
    cx = torch.empty(s[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, s[-1] - mask_size // 2 - is_even)
    dy = torch.arange(s[-2], device=inputs.device).view(1, 1, s[-2], 1) - cy.view(-1, 1, 1, 1)
    dx = torch.arange(s[-1], device=inputs.device).view(1, 1, 1, s[-1]) - cx.view(-1, 1, 1, 1)
    return ((dy >= -(mask_size // 2) + is_even) & (dy <= mask_size // 2) &
            (dx >= -(mask_size // 2) + is_even) & (dx <= mask_size // 2))

def batch_cutmix(inputs, targets, patch_size):
    with torch.no_grad():
        perm = torch.randperm(inputs.shape[0], device='cuda')
        mask = make_random_square_masks(inputs, patch_size)
        if mask is None:
            return inputs, targets
        mixed = torch.where(mask, inputs.index_select(0, perm), inputs)
        portion = float(patch_size ** 2) / (inputs.shape[-2] * inputs.shape[-1])
        labels  = portion * targets.index_select(0, perm) + (1. - portion) * targets
        return mixed, labels

def batch_crop(inputs, crop_size):
    with torch.no_grad():
        mask = make_random_square_masks(inputs, crop_size)
        return torch.masked_select(inputs, mask).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_flip_lr(images, p=0.5):
    with torch.no_grad():
        return torch.where(torch.rand_like(images[:, 0, 0, 0].view(-1, 1, 1, 1)) < p,
                           torch.flip(images, (-1,)), images)

@torch.no_grad()
def get_batches(data_dict, key, batchsize, epoch_fraction=1., cutmix_size=None):
    n = len(data_dict[key]['images'])
    shuffled = torch.randperm(n, device='cuda')
    if epoch_fraction < 1:
        shuffled = shuffled[:batchsize * round(epoch_fraction * shuffled.shape[0] / batchsize)]
        n = shuffled.shape[0]
    if key == 'train':
        images = batch_flip_lr(batch_crop(data_dict[key]['images'], 32))
        images, targets = batch_cutmix(images, data_dict[key]['targets'], patch_size=cutmix_size or 0)
    else:
        images, targets = data_dict[key]['images'], data_dict[key]['targets']
    images = images.to(memory_format=torch.channels_last)
    for idx in range(n // batchsize):
        if (idx + 1) * batchsize <= n:
            yield (images.index_select(0, shuffled[idx * batchsize:(idx + 1) * batchsize]),
                   targets.index_select(0, shuffled[idx * batchsize:(idx + 1) * batchsize]))

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(data_location='data.pt', device='cuda'):
    import torchvision
    from torchvision import transforms
    if not os.path.exists(data_location):
        transform = transforms.Compose([transforms.ToTensor()])
        c10_train = torchvision.datasets.CIFAR10('cifar10/', download=True,  train=True,  transform=transform)
        c10_eval  = torchvision.datasets.CIFAR10('cifar10/', download=False, train=False, transform=transform)
        def _load_all(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=2)
            imgs, lbls = next(iter(loader))
            return imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        train_imgs, train_lbls = _load_all(c10_train)
        eval_imgs,  eval_lbls  = _load_all(c10_eval)
        std, mean = torch.std_mean(train_imgs, dim=(0, 2, 3))
        def norm(x): return (x - mean.view(1,-1,1,1)) / std.view(1,-1,1,1)
        data = {
            'train': {'images': norm(train_imgs).half(), 'targets': F.one_hot(train_lbls).half()},
            'eval':  {'images': norm(eval_imgs).half(),  'targets': F.one_hot(eval_lbls).half()},
        }
        torch.save(data, data_location)
    else:
        data = torch.load(data_location, map_location=device)
    return data

# ── Model factory ──────────────────────────────────────────────────────────────
def make_net(hyp, data):
    scaler = 2.0
    base   = hyp['net']['base_depth']
    depths = {
        'init':   round(scaler ** -1 * base),
        'block1': round(scaler **  0 * base),
        'block2': round(scaler **  2 * base),
        'block3': round(scaler **  3 * base),
    }
    wcd = 3 * hyp['net']['whitening']['kernel_size'] ** 2
    bn_m = hyp['net']['batch_norm_momentum']
    net_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten':     Conv(3, wcd, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0),
            'activation': nn.GELU(),
        }),
        'conv_group_1': ConvGroup(2 * wcd,        depths['block1'], bn_momentum=bn_m),
        'conv_group_2': ConvGroup(depths['block1'], depths['block2'], bn_momentum=bn_m),
        'conv_group_3': ConvGroup(depths['block2'], depths['block3'], bn_momentum=bn_m),
        'pooling':      FastGlobalMaxPooling(),
        'linear':       Linear(depths['block3'], 10, bias=False, temperature=hyp['opt']['scaling_factor']),
    })
    net = SpeedyConvNet(net_dict).to(hyp['misc']['device'])
    net = net.to(memory_format=torch.channels_last)
    net.train().half()

    with torch.no_grad():
        pad = hyp['net']['pad_amount']
        init_whitening_conv(
            net.net_dict['initial_block']['whiten'],
            data['train']['images'].index_select(0, torch.randperm(data['train']['images'].shape[0], device=data['train']['images'].device)),
            num_examples=hyp['net']['whitening']['num_examples'],
            pad_amount=pad,
        )
        for name in net.net_dict.keys():
            if 'conv_group' in name:
                dirac = torch.nn.init.dirac_(torch.empty_like(net.net_dict[name].conv1.weight))
                std_pre, mean_pre = torch.std_mean(net.net_dict[name].conv1.weight.data)
                net.net_dict[name].conv1.weight.data += dirac
                std_post, mean_post = torch.std_mean(net.net_dict[name].conv1.weight.data)
                net.net_dict[name].conv1.weight.data.sub_(mean_post).div_(std_post).mul_(std_pre).add_(mean_pre)
                torch.nn.init.dirac_(net.net_dict[name].conv2.weight)
    return net

# ── Main training function ─────────────────────────────────────────────────────
def run_skeleton(config: dict | None = None, apply_fn=None) -> dict:
    """
    Train SpeedyConvNet (hlb-cifar10) on CIFAR-10.
    apply_fn(net, hyp) -> (net, hyp)  — agent hook to optimize training.
    """
    if config is None:
        config = yaml.safe_load(Path("config.yaml").read_text())

    import copy as _copy
    hyp = _copy.deepcopy(DEFAULT_HYP)
    hyp['misc']['train_epochs'] = config.get('train_epochs', DEFAULT_HYP['misc']['train_epochs'])
    hyp['misc']['data_location'] = config.get('data_location', 'data.pt')

    device = hyp['misc']['device']
    batchsize = config.get('batch_size', 1024)

    # Load GPU-resident dataset
    data = load_data(data_location=hyp['misc']['data_location'], device=device)

    # Pad training images
    if hyp['net']['pad_amount'] > 0:
        data['train']['images'] = F.pad(data['train']['images'],
                                        (hyp['net']['pad_amount'],) * 4, 'reflect')

    # Build model
    net = make_net(hyp, data)

    # ── Agent applies its optimization here ──────────────────────────────────
    if apply_fn is not None:
        try:
            result = apply_fn(net, hyp)
            if isinstance(result, tuple) and len(result) == 2:
                net, hyp = result
            else:
                net = result  # apply_fn returned only net
            print("[skeleton] pytorch_binding applied successfully")
        except Exception as e:
            print(f"[skeleton] pytorch_binding apply failed (falling back to baseline): {e}")

    # ── Optimizers ───────────────────────────────────────────────────────────
    params_non_bias = {'params': [], 'lr': hyp['opt']['non_bias_lr'], 'momentum': 0.85,
                       'nesterov': True, 'weight_decay': hyp['opt']['non_bias_decay'], 'foreach': True}
    params_bias     = {'params': [], 'lr': hyp['opt']['bias_lr'],     'momentum': 0.85,
                       'nesterov': True, 'weight_decay': hyp['opt']['bias_decay'],     'foreach': True}
    for name, p in net.named_parameters():
        if p.requires_grad:
            (params_bias if 'bias' in name else params_non_bias)['params'].append(p)

    opt      = torch.optim.SGD(**params_non_bias)
    opt_bias = torch.optim.SGD(**params_bias)

    num_steps     = len(data['train']['images']) // batchsize
    total_steps   = math.ceil(num_steps * hyp['misc']['train_epochs'])
    ema_start     = math.floor(hyp['misc']['train_epochs']) - hyp['misc']['ema']['epochs']
    ema_decay_val = hyp['misc']['ema']['decay_base'] ** hyp['misc']['ema']['every_n_steps']
    pct_start     = hyp['opt']['percent_start']

    sched_kwargs = dict(pct_start=pct_start, div_factor=1e16,
                        final_div_factor=1./(1e16 * 0.07), total_steps=total_steps,
                        anneal_strategy='linear', cycle_momentum=False)
    lr_sched      = torch.optim.lr_scheduler.OneCycleLR(opt,      max_lr=params_non_bias['lr'], **sched_kwargs)
    lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(opt_bias, max_lr=params_bias['lr'],     **sched_kwargs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')
    loss_batchsize_scaler = 512 / batchsize

    # ── Training loop ─────────────────────────────────────────────────────────
    net_ema       = None
    current_steps = 0
    total_time_s  = 0.0
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    for epoch in range(math.ceil(hyp['misc']['train_epochs'])):
        torch.cuda.synchronize()
        starter.record()
        net.train()

        cutmix_size   = hyp['net']['cutmix_size'] if epoch >= hyp['misc']['train_epochs'] - hyp['net']['cutmix_epochs'] else 0
        epoch_fraction = 1 if epoch + 1 < hyp['misc']['train_epochs'] else hyp['misc']['train_epochs'] % 1

        for inputs, targets in get_batches(data, 'train', batchsize, epoch_fraction, cutmix_size):
            outputs = net(inputs)
            loss = (loss_fn(outputs, targets)
                    .mul(hyp['opt']['loss_scale_scaler'] * loss_batchsize_scaler)
                    .sum()
                    .div(hyp['opt']['loss_scale_scaler']))
            loss.backward()
            opt.step();      opt_bias.step()
            lr_sched.step(); lr_sched_bias.step()
            opt.zero_grad(set_to_none=True)
            opt_bias.zero_grad(set_to_none=True)
            current_steps += 1

            if epoch >= ema_start and current_steps % hyp['misc']['ema']['every_n_steps'] == 0:
                if net_ema is None:
                    net_ema = NetworkEMA(net)
                    continue
                net_ema.update(net, decay=ema_decay_val * (current_steps / total_steps) ** hyp['misc']['ema']['decay_pow'])

        ender.record()
        torch.cuda.synchronize()
        total_time_s += 1e-3 * starter.elapsed_time(ender)

        # ── Per-epoch eval ────────────────────────────────────────────────────
        net.eval()
        acc_list, ema_acc_list = [], []
        with torch.no_grad():
            for inputs, targets in get_batches(data, 'eval', batchsize=2500):
                outputs = net(inputs)
                acc_list.append((outputs.argmax(-1) == targets.argmax(-1)).float().mean())
                if epoch >= ema_start and net_ema is not None:
                    ema_out = net_ema(inputs)
                    ema_acc_list.append((ema_out.argmax(-1) == targets.argmax(-1)).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        ema_acc = torch.stack(ema_acc_list).mean().item() if ema_acc_list else val_acc
        print(f"Epoch {epoch+1:2d}/{math.ceil(hyp['misc']['train_epochs'])}  "
              f"val_acc={val_acc:.4f}  ema_acc={ema_acc:.4f}  time={total_time_s:.2f}s")

    final_acc = ema_acc if ema_acc_list else val_acc
    target_acc  = config.get('target_accuracy',    0.94)
    target_time = config.get('target_wall_time_s', 10.0)

    result = {
        'accuracy':    round(final_acc, 4),
        'wall_time_s': round(total_time_s, 3),
        'passed':      final_acc >= target_acc and total_time_s < target_time,
        'epochs':      math.ceil(hyp['misc']['train_epochs']),
    }
    print(f"\nResult: accuracy={result['accuracy']}  wall_time={result['wall_time_s']}s  passed={result['passed']}")
    return result


if __name__ == "__main__":
    result = run_skeleton()
    import json
    Path("logs/baseline_hlb.json").write_text(json.dumps(result, indent=2))
    print(f"Saved to logs/baseline_hlb.json")
