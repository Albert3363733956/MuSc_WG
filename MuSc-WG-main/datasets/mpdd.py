import os
from enum import Enum
import random

import PIL
import torch
from torchvision import transforms


_CLASSNAMES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MPDDDataset(torch.utils.data.Dataset):
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
        for index, sample in enumerate(full_datasets):
            anomaly_type = os.path.basename(os.path.dirname(sample[2]))
            id_dict.setdefault(anomaly_type, []).append(index)

        sub_id_list = []
        for type_id_list in id_dict.values():
            random.shuffle(type_id_list)
            divided_list = [
                type_id_list[i : i + divide_num]
                for i in range(0, len(type_id_list), divide_num)
            ]
            sub_id_list.extend(
                group[divide_iter]
                for group in divided_list
                if len(group) > divide_iter
            )

        return [full_datasets[index] for index in sub_id_list]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
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

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = sorted(os.listdir(classpath))

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = self._list_image_files(anomaly_path)
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, filename) for filename in anomaly_files
                ]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    maskpaths_per_class[classname][anomaly] = self._build_mask_lookup(
                        anomaly_mask_path
                    )
                else:
                    maskpaths_per_class[classname][anomaly] = {}

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for image_path in imgpaths_per_class[classname][anomaly]:
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        image_stem = os.path.splitext(os.path.basename(image_path))[0]
                        mask_path = maskpaths_per_class[classname][anomaly].get(image_stem)
                        if mask_path is None:
                            raise FileNotFoundError(
                                f"Missing MPDD mask for image: {image_path}"
                            )
                        data_tuple.append(mask_path)
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    @staticmethod
    def _list_image_files(path):
        return sorted(
            filename
            for filename in os.listdir(path)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )

    @staticmethod
    def _build_mask_lookup(path):
        lookup = {}
        for filename in MPDDDataset._list_image_files(path):
            stem = os.path.splitext(filename)[0]
            image_stem = stem[:-5] if stem.endswith("_mask") else stem
            lookup[image_stem] = os.path.join(path, filename)
        return lookup
