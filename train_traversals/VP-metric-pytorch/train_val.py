"""Training and validation loops for the VP classifier."""

import contextlib
import time

import torch


def prepare_batch(inputs, target, device, channels_last=False):
    """Transfer and normalize a uint8 batch using the historical operations."""
    inputs = inputs.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    # This deliberately mirrors ToTensor + Normalize rather than replacing it
    # with a fused affine expression that can round differently.
    inputs.div_(255.0).sub_(0.5).div_(0.5)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    target = target.to(device=device, non_blocking=True)
    return inputs, target


class CudaPrefetchLoader:
    """Overlap the next pinned-memory transfer with current-batch compute."""

    def __init__(self, loader, device, channels_last=False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)

        def preload():
            try:
                host_inputs, host_target = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return prepare_batch(
                    host_inputs,
                    host_target,
                    self.device,
                    self.channels_last,
                )

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            inputs, target = next_batch
            inputs.record_stream(torch.cuda.current_stream(self.device))
            target.record_stream(torch.cuda.current_stream(self.device))
            next_batch = preload()
            yield inputs, target


class MetricAccumulator:
    """Accumulate exact epoch metrics on-device and synchronize only on demand."""

    def __init__(self, device, topk):
        # MPS does not support float64 tensors. The per-batch loss is float32
        # in the historical path, so retaining it here does not affect training.
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        self.top1_correct = torch.zeros((), dtype=torch.int64, device=device)
        self.topk_correct = torch.zeros((), dtype=torch.int64, device=device)
        self.samples = 0
        self.topk = topk

    def update(self, loss, output, target):
        batch_size = target.shape[0]
        self.loss_sum.add_(loss.detach().float(), alpha=batch_size)
        predictions = output.detach().topk(self.topk, dim=1).indices
        correct = predictions.eq(target.unsqueeze(1))
        self.top1_correct.add_(correct[:, 0].sum())
        self.topk_correct.add_(correct.sum())
        self.samples += batch_size

    def values(self):
        if not self.samples:
            raise RuntimeError("cannot report metrics for an empty loader")
        loss = self.loss_sum.cpu().item()
        top1, topk = torch.stack(
            (self.top1_correct, self.topk_correct)
        ).cpu().tolist()
        return {
            "loss": loss / self.samples,
            "accuracy_top1": top1 * 100.0 / self.samples,
            "accuracy_topk": topk * 100.0 / self.samples,
            "topk": self.topk,
            "samples": self.samples,
        }


def _autocast(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _prepared_batches(loader, device, channels_last, cuda_prefetch):
    if cuda_prefetch and device.type == "cuda":
        return CudaPrefetchLoader(loader, device, channels_last)

    def batches():
        for inputs, target in loader:
            yield prepare_batch(inputs, target, device, channels_last)

    return batches()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    amp_dtype=None,
    scaler=None,
    channels_last=False,
    cuda_prefetch=True,
):
    started = time.perf_counter()
    model.train()
    metrics = None

    batches = _prepared_batches(
        train_loader, device, channels_last, cuda_prefetch
    )
    for step, (inputs, target) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            output = model(inputs)
            loss = criterion(output, target)

        if metrics is None:
            metrics = MetricAccumulator(device, min(5, output.shape[1]))
        metrics.update(loss, output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step % 30 == 0:
            current = metrics.values()
            print(
                "Epoch [{0}][{1}/{2}] loss {3:.4f} acc@1 {4:.3f}".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    current["loss"],
                    current["accuracy_top1"],
                )
            )

    result = metrics.values()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


@torch.inference_mode()
def validate(
    val_loader,
    model,
    criterion,
    epoch,
    device,
    amp_dtype=None,
    channels_last=False,
    cuda_prefetch=True,
):
    started = time.perf_counter()
    model.eval()
    metrics = None

    batches = _prepared_batches(
        val_loader, device, channels_last, cuda_prefetch
    )
    for inputs, target in batches:
        with _autocast(device, amp_dtype):
            output = model(inputs)
            loss = criterion(output, target)
        if metrics is None:
            metrics = MetricAccumulator(device, min(5, output.shape[1]))
        metrics.update(loss, output, target)

    result = metrics.values()
    result["elapsed_seconds"] = time.perf_counter() - started
    print(
        "Validation epoch {0}: loss {1:.5f}, acc@1 {2:.3f}, acc@{3} {4:.3f}".format(
            epoch + 1,
            result["loss"],
            result["accuracy_top1"],
            result["topk"],
            result["accuracy_topk"],
        )
    )
    return result
