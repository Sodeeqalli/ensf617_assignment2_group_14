from torchvision import transforms

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
    }