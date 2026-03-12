import torch


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        max_k = max(topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k / targets.size(0)).item())

        return results


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))

        running_loss += loss.item()
        running_top1 += top1
        running_top5 += top5
        total_batches += 1

    return {
        "loss": running_loss / total_batches,
        "top1": running_top1 / total_batches,
        "top5": running_top5 / total_batches,
    }


def evaluate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))

            running_loss += loss.item()
            running_top1 += top1
            running_top5 += top5
            total_batches += 1

    return {
        "loss": running_loss / total_batches,
        "top1": running_top1 / total_batches,
        "top5": running_top5 / total_batches,
    }