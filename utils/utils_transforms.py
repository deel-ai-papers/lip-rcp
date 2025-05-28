import torch
import os

import numpy as np
import torchvision.transforms.v2 as v2
from .datasets import (
    load_tiny_imagenet,
    load_imagenette,
    CustomDataset,
)
from utils.datasets import load_imagenet1k
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomCrop,
    CenterCrop,
    RandomResizedCrop,
    RandomApply,
    RandomRotation,
    AutoAugment,
    RandAugment,
    ColorJitter,
    Normalize,
    ToDtype,
    ToImage,
    MixUp,
)

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2023, 0.1994, 0.2010]
_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STD = [0.2675, 0.2565, 0.2761]
_TINYIMAGENET_MEAN = [0.4802, 0.4481, 0.3975]
_TINYIMAGENET_STD = [0.2302, 0.2265, 0.2262]
_IMAGENETTE_MEAN = [0.4611, 0.4406, 0.4059]
_IMAGENETTE_STD = [0.2686, 0.2618, 0.2752]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class UniformResize(torch.nn.Module):
    def __init__(self, lower: int, upper: int):
        super(UniformResize, self).__init__()
        self.lower = lower
        self.upper = upper + 1

    def forward(self, x):
        size = torch.randint(self.lower, self.upper, size=[]).item()
        return v2.Resize(size)(x)


LIPCONSTANT_PREPROCESS_DATASET = {
    "cifar10": 1 / min(_CIFAR10_STD),
    "cifar100": 1 / min(_CIFAR100_STD),
    "tinyimagenet": 1 / min(_TINYIMAGENET_STD),
    "imagenette": 1 / min(_IMAGENETTE_STD),
    "imagenet": 1 / min(_IMAGENET_STD),
}
dict_normalizations = {
    "cifar10": v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    "cifar100": v2.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    "tinyimagenet": v2.Normalize(mean=_TINYIMAGENET_MEAN, std=_TINYIMAGENET_STD),
    "imagenette": v2.Normalize(mean=_IMAGENETTE_MEAN, std=_IMAGENETTE_STD),
    "imagenet": v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
}


def get_data(
    dataset: str,
    augmentations_prob: list,
    normalize: bool = True,
    p_augmentations_imgs: float = 0.5,
    p_augmentations_mix: float = 0.1,
):
    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "tinyimagenet": 200,
        "imagenette": 10,
        "imagenet": 1000,
    }[dataset.lower()]
    sz_img = {
        "cifar10": (32, 32),
        "cifar100": (32, 32),
        "tinyimagenet": (64, 64),
        "imagenet": (224, 224),
    }[dataset.lower()]

    policy = {
        "cifar10": v2.AutoAugmentPolicy.CIFAR10,
        "cifar100": v2.AutoAugmentPolicy.CIFAR10,
        "tinyimagenet": v2.AutoAugmentPolicy.IMAGENET,
        "imagenet": v2.AutoAugmentPolicy.IMAGENET,
    }[dataset.lower()]

    dict_transforms_prob = {
        "horizontal_flip": v2.RandomHorizontalFlip(p=p_augmentations_imgs),
        "vertical_flip": v2.RandomVerticalFlip(p=p_augmentations_imgs),
        "rotation": v2.RandomApply(
            [
                v2.RandomRotation(degrees=15),
            ],
            p=p_augmentations_imgs,
        ),
        "color_jitter": v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            ],
            p=p_augmentations_imgs,
        ),
        "equalize": v2.RandomEqualize(p=p_augmentations_imgs),
        "crop": v2.RandomApply(
            [
                v2.RandomCrop(
                    size=sz_img,
                    padding=4,
                )
            ],
            p=p_augmentations_imgs,
        ),
        "autoaugment": v2.AutoAugment(policy),
    }
    dict_transforms_mix = {
        "cutmix": v2.RandomApply(
            [v2.CutMix(num_classes=num_classes)], p=p_augmentations_mix
        ),
        "mixup": v2.RandomApply(
            [v2.MixUp(num_classes=num_classes)], p=p_augmentations_mix
        ),
    }

    # Define the transforms
    list_transforms_train = []
    list_transforms_pre = []
    list_transforms_post = []
    list_data_transforms = []

    list_transforms_pre.append(v2.ToImage())

    # Datasets with different image sizes
    if dataset.lower() == "imagenette":
        list_transforms_pre.append(v2.Resize((160, 160)))
    if dataset.lower() == "tinyimagenet":
        list_transforms_pre.append(v2.Resize((64, 64)))

    for aug in augmentations_prob:
        if aug in dict_transforms_prob:
            if aug == "color_jitter":
                assert dataset.lower() not in [
                    "mnist",
                    "fashionmnist",
                ], "Color jitter is only supported for RGB images"
            list_transforms_train.append(dict_transforms_prob[aug])
        elif aug in dict_transforms_mix:
            list_data_transforms.append(dict_transforms_mix[aug])
        else:
            raise ValueError(
                f"Augmentation {aug} not supported yet. Please use one of the following: {list(dict_transforms_prob.keys()) + list(dict_transforms_mix.keys())}"
            )

    list_transforms_post.append(v2.ConvertImageDtype(torch.float32))
    if normalize:
        assert dataset.lower() in dict_normalizations, (
            f"Dataset {dataset} not supported yet."
        )
        list_transforms_post.append(dict_normalizations[dataset.lower()])

    train_transform = v2.Compose(
        list_transforms_pre + list_transforms_train + list_transforms_post
    )
    test_transform = v2.Compose(list_transforms_pre + list_transforms_post)
    data_transform = v2.Compose(list_data_transforms) if list_data_transforms else None

    # Load the dataset
    from torchvision.datasets import CIFAR10, CIFAR100

    if dataset.lower() == "cifar10":
        pre_train_dataset = CIFAR10(
            root=os.environ.get(
                "TORCH_DATASETS"
            ),  # Replace by the path to the CIFAR10 dataset on your machine
            train=True,
            transform=train_transform,
            download=False,
        )
        pre_test_dataset = CIFAR10(
            root=os.environ.get("TORCH_DATASETS"),
            train=False,
            transform=test_transform,
            download=False,
        )

        train_dataset = pre_train_dataset
        val_dataset, cal_dataset, test_dataset = torch.utils.data.random_split(
            pre_test_dataset, [500, 4500, 5000]
        )
    elif dataset.lower() == "cifar100":
        pre_train_dataset = CIFAR100(
            root=os.environ.get(
                "TORCH_DATASETS"
            ),  # Replace by the path to the CIFAR100 dataset on your machine
            train=True,
            transform=train_transform,
            download=False,
        )
        pre_test_dataset = CIFAR100(
            root=os.environ.get("TORCH_DATASETS"),
            train=False,
            transform=test_transform,
            download=False,
        )

        train_dataset = pre_train_dataset
        val_dataset, cal_dataset, test_dataset = torch.utils.data.random_split(
            pre_test_dataset, [500, 4500, 5000]
        )
    elif dataset.lower() == "imagenette":
        # Also assumes that your environment contains the IMAGENETTE variable
        train_data, val_data, cal_data, test_data = load_imagenette(conformal=True)
        train_dataset = CustomDataset(
            train_data[0], train_data[1], transform=train_transform
        )
        val_dataset = CustomDataset(val_data[0], val_data[1], transform=test_transform)
        cal_dataset = CustomDataset(cal_data[0], cal_data[1], transform=test_transform)
        test_dataset = CustomDataset(
            test_data[0], test_data[1], transform=test_transform
        )
    elif dataset.lower() == "tinyimagenet":
        # Also assumes that your environment contains the TINYIMAGENET variable
        train_data, val_data, cal_data, test_data = load_tiny_imagenet(conformal=True)
        train_dataset = CustomDataset(
            train_data[0], train_data[1], transform=train_transform
        )
        val_dataset = CustomDataset(val_data[0], val_data[1], transform=test_transform)
        cal_dataset = CustomDataset(cal_data[0], cal_data[1], transform=test_transform)
        test_dataset = CustomDataset(
            test_data[0], test_data[1], transform=test_transform
        )
    else:
        raise NotImplementedError("Dataset not supported yet.")

    print(
        f"Loading dataset with n_train = {len(train_dataset)}, n_val = {len(val_dataset)}, n_cal = {len(cal_dataset)}, n_test = {len(test_dataset)}"
    )

    return (
        train_dataset,
        val_dataset,
        cal_dataset,
        test_dataset,
        data_transform,
    )


def get_loaders(args):
    if args.dataset != "imagenet":
        from utils.utils_transforms import get_data

        train_dataset, val_dataset, cal_dataset, test_dataset, data_trfm = get_data(
            args.dataset,
            augmentations_prob=[],
            normalize=True,
            p_augmentations_imgs=0,
            p_augmentations_mix=0,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        cal_loader = torch.utils.data.DataLoader(
            cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    else:
        from utils.utils_transforms import UniformResize

        transform = Compose(
            [
                ToImage(),
                Resize((256, 256)),
                CenterCrop(224),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_ds, val_ds = load_imagenet1k(
            False,
            transform,
            transform,
        )
        cal_ds, test_ds = torch.utils.data.random_split(
            val_ds, [len(val_ds) // 2, len(val_ds) - len(val_ds) // 2]
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        cal_loader = torch.utils.data.DataLoader(
            cal_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    return cal_loader, test_loader
