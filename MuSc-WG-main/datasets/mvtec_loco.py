import os
from enum import Enum
import random

import numpy as np
import PIL
import torch
from torchvision import transforms


_CLASSNAMES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class MVTecLOCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        clip_transformer=None,
        k_shot=0,
        random_seed=42,
        divide_num=1,
        divide_iter=0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        if divide_num > 1:
            self.data_to_iterate = self.sub_datasets(
                self.data_to_iterate,
                divide_num,
                divide_iter,
                random_seed,
            )

        if k_shot > 0:
            torch.manual_seed(random_seed)
            if k_shot < len(self.data_to_iterate):
                indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
                self.data_to_iterate = [self.data_to_iterate[i] for i in indices]

        if clip_transformer is None:
            self.transform_img = transforms.Compose(
                [
                    transforms.Resize((resize, resize)),
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform_img = clip_transformer

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)

    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)

        id_dict = {}
        for i, sample in enumerate(full_datasets):
            anomaly_type = os.path.basename(os.path.dirname(sample[2]))
            id_dict.setdefault(anomaly_type, []).append(i)

        sub_id_list = []
        for type_id_list in id_dict.values():
            random.shuffle(type_id_list)
            divide_list = [
                type_id_list[i : i + divide_num]
                for i in range(0, len(type_id_list), divide_num)
            ]
            sub_id_list.extend(
                divide[divide_iter]
                for divide in divide_list
                if len(divide) > divide_iter
            )

        return [full_datasets[idx] for idx in sub_id_list]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_paths = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_paths is not None:
            mask = self.load_mask(mask_paths)
            mask = self.transform_mask(mask) > 0
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def load_mask(self, mask_paths):
        if isinstance(mask_paths, str):
            mask_paths = [mask_paths]

        merged_mask = None
        for mask_path in mask_paths:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = np.asarray(mask)
            if merged_mask is None:
                merged_mask = mask
            else:
                merged_mask = np.maximum(merged_mask, mask)

        return PIL.Image.fromarray(merged_mask.astype(np.uint8))

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            if not os.path.isdir(classpath):
                raise FileNotFoundError(f"MVTec LOCO split directory not found: {classpath}")

            anomaly_types = sorted(
                item
                for item in os.listdir(classpath)
                if os.path.isdir(os.path.join(classpath, item))
            )

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(
                    item
                    for item in os.listdir(anomaly_path)
                    if os.path.isfile(os.path.join(anomaly_path, item))
                    and item.lower().endswith(_IMAGE_EXTENSIONS)
                )
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, filename)
                    for filename in anomaly_files
                ]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    maskpaths_per_class[classname][anomaly] = [
                        self.find_mask_paths(maskpath, anomaly, image_path)
                        for image_path in imgpaths_per_class[classname][anomaly]
                    ]
                else:
                    maskpaths_per_class[classname][anomaly] = [
                        None for _ in imgpaths_per_class[classname][anomaly]
                    ]

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_to_iterate.append(
                        [
                            classname,
                            anomaly,
                            image_path,
                            maskpaths_per_class[classname][anomaly][i],
                        ]
                    )

        return imgpaths_per_class, data_to_iterate

    def find_mask_paths(self, mask_root, anomaly, image_path):
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_dir = os.path.join(mask_root, anomaly, image_stem)
        if os.path.isdir(mask_dir):
            mask_files = sorted(
                item
                for item in os.listdir(mask_dir)
                if os.path.isfile(os.path.join(mask_dir, item))
                and item.lower().endswith(_IMAGE_EXTENSIONS)
            )
            if mask_files:
                return [os.path.join(mask_dir, filename) for filename in mask_files]

        direct_mask_dir = os.path.join(mask_root, anomaly)
        for extension in _IMAGE_EXTENSIONS:
            direct_mask_path = os.path.join(direct_mask_dir, f"{image_stem}{extension}")
            if os.path.isfile(direct_mask_path):
                return [direct_mask_path]

        raise FileNotFoundError(
            f"Mask not found for MVTec LOCO image: {image_path}"
        )
