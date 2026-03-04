import torch


# Compute classification accuracy for a batch
def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)  # predicted class = highest logit
    return (preds == labels).float().mean().item()


def run_one_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None  # training if optimizer is provided
    model.train(train_mode)

    ce = torch.nn.CrossEntropyLoss()  # standard loss for multi-class classification
    total_loss, total_acc, n = 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)           # forward pass
        loss = ce(logits, labels)      # compute loss
        acc = accuracy(logits, labels) # compute accuracy

        if train_mode:
            optimizer.zero_grad()  # clear gradients
            loss.backward()        # backpropagation
            optimizer.step()       # update weights

        total_loss += loss.item()
        total_acc += acc
        n += 1

    # return average loss and accuracy for the epoch
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_confusion_and_wrongs(model, loader, device, idx_to_class):
    model.eval()  # evaluation mode
    num_classes = len(idx_to_class)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # confusion matrix
    wrong_examples = {}  # store one wrong prediction per true class

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        for i in range(labels.size(0)):
            t = labels[i].item()  # true class
            p = preds[i].item()   # predicted class
            cm[t, p] += 1         # update confusion matrix

            if p != t and t not in wrong_examples:
                # store first incorrect example for this true class
                wrong_examples[t] = (p, imgs[i].detach().cpu())

    return cm, wrong_examples