#!/usr/bin/env python3
"""Train the checkpoint CNN with transfer learning, on the same split as the classical models.

    python3 ml/train_cnn.py <dataset dir> [--arch resnet18|mobilenet_v3_small] [--epochs 15]

* starts from ImageNet weights (transfer learning) and replaces the last layer with one output per
  class: normal, blocked_exit, obstructed_walkway, open_door
* trains on the TRAIN split with light augmentation (crop, flip, brightness/contrast), class-weighted
  loss so the rarer hazard classes count as much as 'normal'
* after every epoch scores the VALIDATION split and keeps the best epoch (early stopping on macro F1)
* scores the TEST split once with the best weights
Writes <dataset>/results/cnn_<arch>.pt, cnn_<arch>.onnx (with --onnx, for TensorRT on the Jetson) and a
row in results/metrics.csv. Uses the Apple GPU (mps) or CUDA when available.
Needs: pip install torch torchvision   (the ImageNet weights download once, ~45 MB)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import CLASSES, by_split, confusion, line, load, print_confusion, save_result, scores  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--arch', default='resnet18', choices=['resnet18', 'mobilenet_v3_small'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=5, help='stop after this many epochs without a better val score')
    ap.add_argument('--size', type=int, default=224, help='input size in px')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--no-pretrained', action='store_true', help='train from scratch (to show what transfer learning buys)')
    ap.add_argument('--onnx', action='store_true', help='also export ONNX for TensorRT')
    ap.add_argument('--seed', type=int, default=505)
    a = ap.parse_args()

    try:                     # python.org Python on macOS ships without CA certificates -> weight download fails
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    import torch
    import torchvision
    from torch import nn

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    sp = by_split(load(a.dataset))
    for s in sp:
        print(f'{s:5s} {len(sp[s]):5d} frames, classes {[CLASSES[c] for c in sorted({r["y"] for r in sp[s]})]}')
    if not sp['train'] or not sp['val'] or not sp['test']:
        raise SystemExit('need frames in train, val and test')

    def read(r):
        bgr = cv2.imread(os.path.join(a.dataset, r['file_rgb']))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    imgs = {s: [read(r) for r in sp[s]] for s in sp}           # small dataset: keep it in memory
    ys = {s: np.array([r['y'] for r in sp[s]]) for s in sp}

    def tensor(batch_imgs, train):
        out = []
        for im in batch_imgs:
            if train:
                h, w = im.shape[:2]
                sc = np.random.uniform(0.8, 1.0)
                ch, cw = int(h * sc), int(w * sc)
                y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
                im = im[y0:y0 + ch, x0:x0 + cw]
                if np.random.rand() < 0.5:
                    im = im[:, ::-1]
                im = np.clip(im.astype(np.float32) * np.random.uniform(0.7, 1.3) + np.random.uniform(-25, 25), 0, 255)
            im = cv2.resize(np.ascontiguousarray(im).astype(np.float32), (a.size, a.size)) / 255.0
            out.append(((im - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(out)).float()

    weights = None if a.no_pretrained else 'DEFAULT'
    if a.arch == 'resnet18':
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    else:
        net = torchvision.models.mobilenet_v3_small(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(CLASSES))
    net = net.to(dev)

    counts = np.bincount(ys['train'], minlength=len(CLASSES)).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / (counts > 0).sum(), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def predict(images):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), a.batch):
                out.append(net(tensor(images[i:i + a.batch], False).to(dev)).argmax(1).cpu().numpy())
        return np.concatenate(out)

    os.makedirs(os.path.join(a.dataset, 'results'), exist_ok=True)
    ckpt = os.path.join(a.dataset, 'results', f'cnn_{a.arch}.pt')
    best, bad = -1.0, 0
    print(f'training {a.arch} ({"ImageNet weights" if weights else "from scratch"}) on {dev}, '
          f'{len(ys["train"])} train frames, class weights {np.round(w, 2).tolist()}')
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        order = np.random.permutation(len(ys['train']))
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            xb = tensor([imgs['train'][j] for j in idx], True).to(dev)
            yb = torch.from_numpy(ys['train'][idx]).long().to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        v = scores(ys['val'], predict(imgs['val']))
        mark = ''
        if v['macro_f1'] > best:
            best, bad, mark = v['macro_f1'], 0, '  <- best, saved'
            torch.save(net.state_dict(), ckpt)
        else:
            bad += 1
        print(f'epoch {ep:2d}  loss {tot / len(order):.3f}  val macro F1 {v["macro_f1"]:.2f}  '
              f'hazard recall {v["hazard_recall"]:.2f}  ({time.time() - t0:.0f} s){mark}')
        if bad >= a.patience:
            print(f'early stop: no better validation score for {a.patience} epochs')
            break

    net.load_state_dict(torch.load(ckpt, map_location=dev))
    pred = predict(imgs['test'])
    t = scores(ys['test'], pred)
    name = f'CNN {a.arch}' + (' (from scratch)' if a.no_pretrained else '')
    params = f'{a.size} px, best val macro F1 {best:.2f}, {"ImageNet" if weights else "no"} pretraining'
    print('\nTEST (held-out checkpoint and groups)')
    print(line(name, t))
    cm = confusion(ys['test'], pred)
    print_confusion(cm)
    save_result(a.dataset, name, 'test', t, params, cm)

    # latency on this machine (the Jetson number comes from the TensorRT engine)
    net.eval()
    x = tensor(imgs['test'][:1], False).to(dev)
    with torch.no_grad():
        for _ in range(3):
            net(x)
        t0 = time.time()
        for _ in range(20):
            net(x)
        if dev == 'mps':
            torch.mps.synchronize()
    print(f'inference on {dev}: {1000 * (time.time() - t0) / 20:.1f} ms per frame')
    if a.onnx:
        path = ckpt.replace('.pt', '.onnx')
        try:
            torch.onnx.export(net.cpu(), torch.zeros(1, 3, a.size, a.size), path, input_names=['image'],
                              output_names=['logits'], opset_version=13, dynamo=False)
            print(f'ONNX: {path}   (Jetson: trtexec --onnx={os.path.basename(path)} --saveEngine=cnn.engine --fp16)')
        except ImportError as e:
            print(f'ONNX export skipped ({e}); install with:  pip install onnx onnxscript')
    print(f'results: {os.path.join(a.dataset, "results", "metrics.csv")}')


if __name__ == '__main__':
    main()
