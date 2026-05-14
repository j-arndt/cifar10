"""
tinygrad hlb_cifar10 skeleton — MOAB Crucible baseline.
The agent's apply_fn(hyp, env) -> (hyp, env) injects optimizations.
Wall time and accuracy are returned. The agents figure out how to go fast.
"""
import os, random, time
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

def run_skeleton(config: dict | None = None, apply_fn=None) -> dict:
    if config is None:
        config = yaml.safe_load(Path("config.yaml").read_text())

    import copy
    hyp = {
        'seed': 201,
        'opt': {
            'bias_lr':            1.76 * 58 / 512,
            'non_bias_lr':        1.76 / 512,
            'bias_decay':         1.08 * 6.45e-4 * 512 / 58,
            'non_bias_decay':     1.08 * 6.45e-4 * 512,
            'final_lr_ratio':     0.025,
            'initial_div_factor': 1e6,
            'label_smoothing':    0.20,
            'momentum':           0.85,
            'percent_start':      0.23,
            'loss_scale_scaler':  1./128,
        },
        'net': {
            'kernel_size':  2,
            'cutmix_size':  3,
            'cutmix_steps': 499,
            'pad_amount':   2,
        },
        'ema': {
            'steps':        399,
            'decay_base':   .95,
            'decay_pow':    1.6,
            'every_n_steps': 5,
        },
    }

    # env_vars the agent can tune: BEAM, WINO, BS, STEPS, EMA, etc.
    env = {
        'BEAM':         config.get('beam', 0),
        'WINO':         config.get('wino', 0),
        'BS':           config.get('batch_size', 512),
        'STEPS':        config.get('steps', 1000),
        'EMA':          config.get('ema', 1),
        'SYNCBN':       '0',
        'RANDOM_CROP':  '1',
        'RANDOM_FLIP':  '1',
        'CUTMIX':       '1',
    }

    # ── Agent hook ────────────────────────────────────────────────────────────
    if apply_fn is not None:
        try:
            result = apply_fn(hyp, env)
            if isinstance(result, tuple) and len(result) == 2:
                hyp, env = result
            print("[skeleton] tinygrad_binding applied successfully")
        except Exception as e:
            print(f"[skeleton] tinygrad_binding failed (falling back to baseline): {e}")

    # Apply env vars before importing tinygrad (BEAM etc must be set at import)
    for k, v in env.items():
        os.environ[k] = str(v)

    # ── tinygrad imports (after env vars set) ─────────────────────────────────
    from tinygrad import Tensor, TinyJit, Device, GlobalCounters, dtypes
    from tinygrad.helpers import getenv, BEAM, WINO, Context, colored, prod
    from tinygrad.nn import optim
    from tinygrad.nn.state import get_state_dict
    from tinygrad.nn.datasets import cifar

    BS    = getenv("BS", 512)
    STEPS = getenv("STEPS", 1000)

    cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    cifar_std  = [0.24703225141799082, 0.24348516474564,   0.26158783926049628]

    # ── Model ─────────────────────────────────────────────────────────────────
    class UnsyncedBatchNorm:
        def __init__(self, sz, eps=1e-5, momentum=0.1):
            self.eps, self.momentum = eps, momentum
            self.weight = Tensor.ones(sz, dtype=dtypes.float32)
            self.bias   = Tensor.zeros(sz, dtype=dtypes.float32)
            self.running_mean = Tensor.zeros(1, sz, dtype=dtypes.float32, requires_grad=False)
            self.running_var  = Tensor.ones(1,  sz, dtype=dtypes.float32, requires_grad=False)
            self.weight.requires_grad = False
            self.bias.requires_grad   = True

        def __call__(self, x):
            xr = x.reshape(1, -1, *x.shape[1:]).cast(dtypes.float32)
            if Tensor.training:
                batch_mean = xr.mean(axis=(1, 3, 4))
                y = xr - batch_mean.detach().reshape(1, 1, -1, 1, 1)
                batch_var = (y * y).mean(axis=(1, 3, 4))
                batch_invstd = batch_var.add(self.eps).pow(-0.5)
                self.running_mean.assign((1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype))
                batch_var_adj = prod(y.shape[1:]) / (prod(y.shape[1:]) - y.shape[2])
                self.running_var.assign((1 - self.momentum) * self.running_var + self.momentum * batch_var_adj * batch_var.detach().cast(self.running_var.dtype))
            else:
                batch_mean = self.running_mean
                batch_invstd = self.running_var.reshape(1, 1, -1, 1, 1).expand(xr.shape).add(self.eps).rsqrt()
            ret = xr.batchnorm(
                self.weight.reshape(1, -1).expand((1, -1)),
                self.bias.reshape(1, -1).expand((1, -1)),
                batch_mean, batch_invstd, axis=(0, 2))
            return ret.reshape(x.shape).cast(x.dtype)

    class ConvGroup:
        def __init__(self, channels_in, channels_out):
            from tinygrad import nn as tnn
            self.conv1 = tnn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
            self.conv2 = tnn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
            self.norm1 = UnsyncedBatchNorm(channels_out, eps=1e-12, momentum=0.85)
            self.norm2 = UnsyncedBatchNorm(channels_out, eps=1e-12, momentum=0.85)

        def __call__(self, x):
            x = self.conv1(x).max_pool2d(2).float()
            x = self.norm1(x).cast(dtypes.default_float).quick_gelu()
            residual = x
            x = self.conv2(x).float()
            x = self.norm2(x).cast(dtypes.default_float).quick_gelu()
            return x + residual

    def _whitening_weights(X, kernel_size=2):
        def _patches(data, ks):
            h, w = ks, ks
            c = data.shape[1]
            return np.lib.stride_tricks.sliding_window_view(
                data, window_shape=(h, w), axis=(2, 3)
            ).transpose((0, 3, 2, 1, 4, 5)).reshape(-1, c, h, w)
        patches = _patches(X.cast(dtypes.float32).numpy(), kernel_size).astype(np.float32)
        n, c, h, w = patches.shape
        flat = patches.reshape(n, c * h * w)
        cov  = (flat.T @ flat) / (n - 1)
        Lam, V = np.linalg.eigh(cov)
        Lam, V = np.flip(Lam, 0), np.flip(V.T.reshape(c * h * w, c, h, w), 0)
        W = V / np.sqrt(Lam + 1e-2)[:, None, None, None]
        return Tensor(W.astype(np.float32), requires_grad=False).cast(dtypes.default_float)

    class SpeedyResNet:
        def __init__(self, W):
            self.whitening = W
            from tinygrad import nn as tnn
            self.net = [
                tnn.Conv2d(12, 32, kernel_size=1, bias=False),
                lambda x: x.quick_gelu(),
                ConvGroup(32,  64),
                ConvGroup(64,  256),
                ConvGroup(256, 512),
                lambda x: x.max((2, 3)),
                tnn.Linear(512, 10, bias=False),
                lambda x: x / 9.,
            ]

        def __call__(self, x, training=True):
            fwd = lambda x: x.conv2d(self.whitening).pad((1, 0, 0, 1)).sequential(self.net)
            return fwd(x) if training else (fwd(x) + fwd(x[..., ::-1])) / 2.

    class ModelEMA:
        def __init__(self, W, net):
            self.net_ema = SpeedyResNet(W)
            for ep, np_ in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).values()):
                ep.requires_grad = False
                ep.assign(np_.numpy())

        @TinyJit
        def update(self, net, decay):
            for ep, (name, p) in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).items()):
                if "num_batches_tracked" not in name and "running" not in name:
                    ep.assign(ep.detach() * decay + p.detach() * (1. - decay)).realize()

    # ── Data ──────────────────────────────────────────────────────────────────
    Tensor.manual_seed(hyp['seed'])
    random.seed(hyp['seed'])

    X_train, Y_train, X_test, Y_test = cifar()
    Y_train, Y_test = Y_train.one_hot(10), Y_test.one_hot(10)

    def normalize(x):
        x = x.cast(dtypes.float32) / 255.0
        x = x.reshape((-1, 3, 32, 32))
        x = x - Tensor(cifar_mean, dtype=dtypes.float32).reshape(1, 3, 1, 1)
        x = x / Tensor(cifar_std,  dtype=dtypes.float32).reshape(1, 3, 1, 1)
        return x

    X_train, X_test = normalize(X_train), normalize(X_test)

    W = _whitening_weights(X_train, kernel_size=hyp['net']['kernel_size'])

    # Pad training data
    def pad_reflect(X, size=2):
        X = X[..., :, 1:size+1].flip(-1).cat(X, X[..., :, -(size+1):-1].flip(-1), dim=-1)
        X = X[..., 1:size+1, :].flip(-2).cat(X, X[..., -(size+1):-1, :].flip(-2), dim=-2)
        return X

    X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])
    X_train = X_train.cast(dtypes.default_float)
    Y_train  = Y_train.cast(dtypes.default_float)
    X_test   = X_test.cast(dtypes.default_float)
    Y_test   = Y_test.cast(dtypes.default_float)

    # ── Model & optimizers ────────────────────────────────────────────────────
    model = SpeedyResNet(W)
    params = get_state_dict(model)
    p_bias     = [v for k, v in params.items() if v.requires_grad is not False and 'bias'     in k]
    p_non_bias = [v for k, v in params.items() if v.requires_grad is not False and 'bias' not in k]

    opt_bias     = optim.SGD(p_bias,     lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
    opt_non_bias = optim.SGD(p_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])

    # OneCycleLR (self-contained, no extra dependency)
    def _one_cycle_lr(step, max_lr, pct_start, div_factor, final_div_factor, total_steps):
        if step < int(total_steps * pct_start):
            pct = step / int(total_steps * pct_start)
            return max_lr * (1 / div_factor + pct * (1 - 1 / div_factor))
        else:
            pct = (step - int(total_steps * pct_start)) / (total_steps - int(total_steps * pct_start))
            return max_lr * (1 - pct * (1 - 1 / final_div_factor))

    def set_lr(opt, lr):
        for pg in opt.param_groups if hasattr(opt, 'param_groups') else []:
            pg['lr'] = lr
        # tinygrad SGD stores lr as a tensor
        if hasattr(opt, 'lr'):
            opt.lr.assign(Tensor([lr], dtype=dtypes.float32))

    # ── Augmentation ──────────────────────────────────────────────────────────
    def make_square_mask(shape, mask_size):
        BS_, _, H, W = shape
        lx = Tensor.randint(BS_, low=0, high=W - mask_size).reshape(BS_, 1, 1, 1)
        ly = Tensor.randint(BS_, low=0, high=H - mask_size).reshape(BS_, 1, 1, 1)
        ix = Tensor.arange(W, dtype=dtypes.int32).reshape(1, 1, 1, W)
        iy = Tensor.arange(H, dtype=dtypes.int32).reshape(1, 1, H, 1)
        return (ix >= lx) * (ix < (lx + mask_size)) * (iy >= ly) * (iy < (ly + mask_size))

    def random_crop(X, crop_size=32):
        BS_, _, H, W = X.shape
        lx = Tensor.randint(BS_, low=0, high=W - crop_size).reshape(BS_, 1, 1, 1)
        ly = Tensor.randint(BS_, low=0, high=H - crop_size).reshape(BS_, 1, 1, 1)
        ix = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, 1, crop_size)
        iy = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, crop_size, 1)
        return X.gather(-1, (lx + ix).expand(-1, 3, H, -1)).gather(-2, ((ly + iy).expand(-1, 3, crop_size, crop_size)))

    @TinyJit
    def augment(X, Y):
        perms = Tensor.randperm(X.shape[0], device=X.device)
        X = random_crop(X, 32)
        X = (Tensor.rand(X.shape[0], 1, 1, 1) < 0.5).where(X.flip(-1), X).contiguous()
        Xp, Yp = X[perms], Y[perms]
        mask = make_square_mask(X.shape, hyp['net']['cutmix_size'])
        Xcm  = mask.where(Xp, X)
        mix  = float(hyp['net']['cutmix_size'] ** 2) / (32 * 32)
        Ycm  = mix * Yp + (1 - mix) * Y
        return X, Y, Xcm, Ycm

    def cross_entropy(x, y, label_smoothing=0.0):
        d = y.shape[1]
        y = (1 - label_smoothing) * y + label_smoothing / d
        return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

    @TinyJit
    def train_step(model, opt_b, opt_nb, X, Y):
        out  = model(X, training=True)
        ls   = 512 / BS
        loss = cross_entropy(out, Y, label_smoothing=hyp['opt']['label_smoothing'])
        loss = loss * (hyp['opt']['loss_scale_scaler'] * ls) / hyp['opt']['loss_scale_scaler']
        opt_b.zero_grad(); opt_nb.zero_grad()
        loss.backward()
        opt_b.step(); opt_nb.step()
        return loss.realize()

    @TinyJit
    def eval_step(model, X, Y):
        out     = model(X, training=False)
        correct = out.argmax(axis=1) == Y.argmax(axis=1)
        return correct.realize()

    # ── Training loop ─────────────────────────────────────────────────────────
    model_ema  = None
    proj_decay = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
    step       = 0
    eval_acc   = 0.0

    # Shuffle once per pass
    def get_batches(X, Y, bs, train=True):
        n   = X.shape[0]
        idx = list(range(0, (n // bs) * bs, bs))
        if train:
            random.shuffle(idx)
        for i in idx:
            yield X[i:i+bs], Y[i:i+bs]

    t0 = time.monotonic()

    with Tensor.train():
        for X_b, Y_b in get_batches(X_train, Y_train, BS, train=True):
            if step >= STEPS:
                break

            # LR schedule
            lr_nb = _one_cycle_lr(step, hyp['opt']['non_bias_lr'], hyp['opt']['percent_start'],
                                   hyp['opt']['initial_div_factor'],
                                   1./(hyp['opt']['initial_div_factor'] * hyp['opt']['final_lr_ratio']), STEPS)
            lr_b  = _one_cycle_lr(step, hyp['opt']['bias_lr'], hyp['opt']['percent_start'],
                                   hyp['opt']['initial_div_factor'],
                                   1./(hyp['opt']['initial_div_factor'] * hyp['opt']['final_lr_ratio']), STEPS)
            if hasattr(opt_non_bias, 'lr'): opt_non_bias.lr.assign(Tensor([lr_nb]))
            if hasattr(opt_bias,     'lr'): opt_bias.lr.assign(Tensor([lr_b]))

            X_aug, Y_aug, Xcm, Ycm = augment(X_b, Y_b)
            Xin = Xcm if (step >= hyp['net']['cutmix_steps'] and getenv("CUTMIX", 1)) else X_aug
            Yin = Ycm if (step >= hyp['net']['cutmix_steps'] and getenv("CUTMIX", 1)) else Y_aug

            loss = train_step(model, opt_bias, opt_non_bias, Xin, Yin)

            # EMA
            if getenv("EMA", 1) and step > hyp['ema']['steps'] and (step + 1) % hyp['ema']['every_n_steps'] == 0:
                if model_ema is None:
                    model_ema = ModelEMA(W, model)
                model_ema.update(model, Tensor([proj_decay * (step / STEPS) ** hyp['ema']['decay_pow']]))

            if step % 100 == 0:
                print(f"step {step:4d}/{STEPS}  loss={loss.numpy():.4f}  lr={lr_nb:.6f}  t={time.monotonic()-t0:.1f}s")

            step += 1

    wall_time = time.monotonic() - t0

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_net = model_ema.net_ema if model_ema is not None else model
    corrects = []
    for Xb, Yb in get_batches(X_test, Y_test, 500, train=False):
        corrects.extend(eval_step(eval_net, Xb, Yb).numpy().tolist())
    eval_acc = sum(corrects) / len(corrects)

    target_acc  = config.get('target_accuracy', 0.94)
    target_time = config.get('target_wall_time_s', 1.0)
    result = {
        'accuracy':    round(eval_acc, 4),
        'wall_time_s': round(wall_time, 3),
        'passed':      eval_acc >= target_acc and wall_time < target_time,
    }
    print(f"\nResult: accuracy={result['accuracy']}  wall_time={result['wall_time_s']}s  passed={result['passed']}")
    return result


if __name__ == "__main__":
    result = run_skeleton()
    import json
    Path("logs").mkdir(exist_ok=True)
    Path("logs/baseline_tinygrad.json").write_text(json.dumps(result, indent=2))
