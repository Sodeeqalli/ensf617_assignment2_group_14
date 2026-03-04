import torch


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def run_one_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    ce = torch.nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = ce(logits, labels)
        acc = accuracy(logits, labels)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_confusion_and_wrongs(model, loader, device, idx_to_class):
    model.eval()
    num_classes = len(idx_to_class)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    wrong_examples = {}  # true_idx -> (pred_idx, batch_image_tensor)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        for i in range(labels.size(0)):
            t = labels[i].item()
            p = preds[i].item()
            cm[t, p] += 1

            if p != t and t not in wrong_examples:
                # store CPU image tensor for later saving/plotting
                wrong_examples[t] = (p, imgs[i].detach().cpu())

    return cm, wrong_examples