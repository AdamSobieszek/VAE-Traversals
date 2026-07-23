"""Training and validation loops for the VP classifier."""

import time

import torch

from utils import AverageMeter, accuracy


def _as_float(value):
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def _metrics(losses, top1, topk, k, elapsed):
    return {
        "loss": _as_float(losses.avg),
        "accuracy_top1": _as_float(top1.avg),
        "accuracy_topk": _as_float(topk.avg),
        "topk": k,
        "samples": losses.count,
        "elapsed_seconds": elapsed,
    }


def train(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    started = time.time()
    model.train()

    for step, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(inputs)
        loss = criterion(output, target)

        k = min(5, output.shape[1])
        prec1, preck = accuracy(output.detach(), target, topk=(1, k))
        losses.update(_as_float(loss), inputs.size(0))
        top1.update(_as_float(prec1), inputs.size(0))
        topk.update(_as_float(preck), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(
                "Epoch [{0}][{1}/{2}] loss {3:.4f} acc@1 {4:.3f}".format(
                    epoch + 1, step, len(train_loader), losses.avg, top1.avg
                )
            )

    return _metrics(losses, top1, topk, k, time.time() - started)


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    started = time.time()
    model.eval()

    for inputs, target in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(inputs)
        loss = criterion(output, target)

        k = min(5, output.shape[1])
        prec1, preck = accuracy(output, target, topk=(1, k))
        losses.update(_as_float(loss), inputs.size(0))
        top1.update(_as_float(prec1), inputs.size(0))
        topk.update(_as_float(preck), inputs.size(0))

    metrics = _metrics(losses, top1, topk, k, time.time() - started)
    print(
        "Validation epoch {0}: loss {1:.5f}, acc@1 {2:.3f}, acc@{3} {4:.3f}".format(
            epoch + 1,
            metrics["loss"],
            metrics["accuracy_top1"],
            metrics["topk"],
            metrics["accuracy_topk"],
        )
    )
    return metrics
