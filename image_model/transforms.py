from torchvision import transforms

# Standard normalization values used for ImageNet pretrained models
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    # Define data augmentation for training and standard preprocessing for validation/test
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # random crop + scale for augmentation
            transforms.RandomHorizontalFlip(),                     # random flip to increase data diversity
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # color augmentation
            transforms.RandomRotation(10),                         # small rotation for robustness
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),     # normalize to match pretrained backbone
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),               # deterministic resize for validation
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((img_size, img_size)),               # same preprocessing as validation
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
    }