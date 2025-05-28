from PIL import Image
import pandas as pd
import numpy as np

import os
import torch


NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "imagenette": 10,
    "imagenet": 1000,
}
INPUT_SHAPES = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "tinyimagenet": (3, 64, 64),
    "imagenette": (3, 160, 160),
    "imagenet": (3, 224, 224),
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(idx + 1)
        if self.transform:
            img = self.transform(img)
        return img, label


class HuggingFace2Torch(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if self.transform:
            img = self.transform(img)
        return img, label


def load_imagenet1k(conformal: bool, train_transform, test_transform, seed: int = 0):
    from datasets import load_dataset

    train_dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
    val_dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")

    if not conformal:
        train_dataset = HuggingFace2Torch(train_dataset, train_transform)
        val_dataset = HuggingFace2Torch(val_dataset, test_transform)
        return train_dataset, val_dataset

    else:
        import random

        idx = np.arange(50_000)
        random.seed(seed)
        random.shuffle(idx)
        cal_idx, val_idx, test_idx = (
            idx[:20_000],
            idx[20_000:30_000],
            idx[30_000:50_000],
        )
        train_dataset = train_dataset
        cal_data = val_dataset[cal_idx]
        val_data = val_dataset[val_idx]
        test_data = val_dataset[test_idx]
        train_dataset = HuggingFace2Torch(train_dataset, train_transform)
        val_data = HuggingFace2Torch(val_data, test_transform)
        cal_data = HuggingFace2Torch(cal_data, test_transform)
        test_data = HuggingFace2Torch(test_data, test_transform)
        return train_dataset, val_data, cal_data, test_data


def load_tiny_imagenet(conformal: bool, seed: int = 0):
    from sklearn.preprocessing import LabelEncoder
    import random

    dataset_path = os.environ.get(
        "TINY_IMAGENET"
    )  # Replace by the path to the Tiny ImageNet dataset on your machine
    if dataset_path is None:
        raise ValueError("Please set the TINY_IMAGENET environment variable")

    dir_train = os.path.join(dataset_path, "train")
    dir_val = os.path.join(dataset_path, "val")
    dir_test = os.path.join(dataset_path, "test")

    classes = os.listdir(dir_train)  # Get the labels

    encoder_labels = LabelEncoder()
    encoder_labels.fit(classes)

    img_paths = []
    labels = []

    for c in classes:
        list_imgs_ctr = os.listdir(os.path.join(dir_train, c, "images"))
        img_paths_ctr = [
            os.path.join(dir_train, c, "images", img) for img in list_imgs_ctr
        ]
        labels_ctr = [c] * len(list_imgs_ctr)
        list_imgs_cva = os.listdir(os.path.join(dir_val, c, "images"))
        img_paths_cva = [
            os.path.join(dir_val, c, "images", img) for img in list_imgs_cva
        ]
        labels_cva = [c] * len(list_imgs_cva)
        list_imgs_cte = os.listdir(os.path.join(dir_test, c, "images"))
        img_paths_cte = [
            os.path.join(dir_test, c, "images", img) for img in list_imgs_cte
        ]
        labels_cte = [c] * len(list_imgs_cte)
        img_paths += img_paths_ctr + img_paths_cva + img_paths_cte
        labels += labels_ctr + labels_cva + labels_cte

    if len(img_paths) < 110_000:
        raise ValueError("Mistaken dataset size")

    img_paths = np.array(img_paths)
    labels = encoder_labels.transform(labels)
    labels = np.array(labels)

    if conformal:
        idx = np.arange(len(img_paths))
        random.seed(seed)
        random.shuffle(idx)
        train_idx, cal_idx, val_idx, test_idx = (
            idx[:100_000],
            idx[100_000:104_500],
            idx[104_500:105_000],
            idx[105_000:],
        )
        train_data = (img_paths[train_idx], labels[train_idx])
        cal_data = (img_paths[cal_idx], labels[cal_idx])
        val_data = (img_paths[val_idx], labels[val_idx])
        test_data = (img_paths[test_idx], labels[test_idx])
        return train_data, val_data, cal_data, test_data
    else:
        idx = np.arange(len(img_paths))
        random.seed(seed)
        random.shuffle(idx)
        train_idx, test_idx = idx[:100_000], idx[100_000:]
        train_data = (img_paths[train_idx], labels[train_idx])
        test_data = (img_paths[test_idx], labels[test_idx])
        return train_data, test_data


def load_imagenette(conformal: bool, seed: int = 0):
    from sklearn.preprocessing import LabelEncoder
    import random

    dataset_path = os.environ.get(
        "IMAGENETTE"
    )  # Replace by the path to the Imagenette dataset on your machine
    if dataset_path is None:
        raise ValueError("Please set the IMAGENETTE environment variable")

    dir_train = os.path.join(dataset_path, "train")
    dir_val = os.path.join(dataset_path, "val")

    classes = os.listdir(dir_train)  # Get the labels
    classes.remove(".DS_Store")

    encoder_labels = LabelEncoder()
    encoder_labels.fit(classes)

    imgs_paths = []
    labels = []

    for c in classes:
        list_imgs_c = os.listdir(os.path.join(dir_train, c))
        img_paths_c = [
            os.path.join(dir_train, c, img) for img in list_imgs_c if ".JPEG" in img
        ]
        labels_c = [c] * len(list_imgs_c)
        imgs_paths += img_paths_c
        labels += labels_c
        list_imgs_c = os.listdir(os.path.join(dir_val, c))
        img_paths_c = [
            os.path.join(dir_val, c, img) for img in list_imgs_c if ".JPEG" in img
        ]
        labels_c = [c] * len(list_imgs_c)
        imgs_paths += img_paths_c
        labels += labels_c

    imgs_paths = np.array(imgs_paths)
    labels = encoder_labels.transform(labels)

    print("Num samples:", len(imgs_paths))
    if conformal:
        idx = np.arange(len(imgs_paths))
        random.seed(seed)
        random.shuffle(idx)
        train_idx, val_idx, cal_idx, test_idx = (
            idx[:10_000],
            idx[10_000:10_500],
            idx[10_500:12_000],
            idx[12_000:],
        )
        train_data = (imgs_paths[train_idx], labels[train_idx])
        val_data = (imgs_paths[val_idx], labels[val_idx])
        cal_data = (imgs_paths[cal_idx], labels[cal_idx])
        test_data = (imgs_paths[test_idx], labels[test_idx])
        return train_data, val_data, cal_data, test_data
    else:
        idx = np.arange(len(imgs_paths))
        random.seed(seed)
        random.shuffle(idx)
        train_idx, test_idx = idx[:10_000], idx[10_000:]
        train_data = (imgs_paths[train_idx], labels[train_idx])
        test_data = (imgs_paths[test_idx], labels[test_idx])
        return train_data, test_data
