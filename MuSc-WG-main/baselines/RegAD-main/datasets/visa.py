import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]


def _load_meta(dataset_path):
    meta_path = os.path.join(dataset_path, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'VisA meta.json not found: {meta_path}')
    with open(meta_path, 'r') as f:
        return json.load(f)


def _resolve_path(dataset_path, relative_path):
    if not relative_path:
        return None
    return os.path.normpath(os.path.join(dataset_path, relative_path))


def _normal_images(dataset_path, class_name, split='train'):
    meta = _load_meta(dataset_path)
    return [
        _resolve_path(dataset_path, item['img_path'])
        for item in meta.get(split, {}).get(class_name, [])
        if int(item.get('anomaly', 0)) == 0
    ]


class FSAD_Dataset_train(Dataset):
    def __init__(
            self,
            dataset_path='../data/visa',
            class_name='candle',
            is_train=True,
            resize=256,
            shot=2,
            batch=32,
    ):
        meta = _load_meta(dataset_path)
        class_pool = sorted(meta.get('train', {}).keys())
        assert class_name in class_pool, 'class_name: {}, should be in {}'.format(class_name, class_pool)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        self.batch = batch
        self.query_dir, self.support_dir = self.load_dataset_folder()
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), getattr(Image, 'Resampling', Image).LANCZOS),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        query_list, support_list = self.query_dir[idx], self.support_dir[idx]
        query_img = None
        support_sub_img = None
        support_img = None

        for i in range(len(query_list)):
            image = Image.open(query_list[i]).convert('RGB')
            image = self.transform_x(image).unsqueeze(dim=0)
            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image], dim=0)

            for k in range(self.shot):
                image = Image.open(support_list[i][k]).convert('RGB')
                image = self.transform_x(image).unsqueeze(dim=0)
                if support_sub_img is None:
                    support_sub_img = image
                else:
                    support_sub_img = torch.cat([support_sub_img, image], dim=0)

            support_sub_img = support_sub_img.unsqueeze(dim=0)
            if support_img is None:
                support_img = support_sub_img
            else:
                support_img = torch.cat([support_img, support_sub_img], dim=0)
            support_sub_img = None

        mask = torch.zeros([self.batch, self.resize, self.resize])
        return query_img, support_img, mask

    def __len__(self):
        return len(self.query_dir)

    def shuffle_dataset(self):
        self.query_dir, self.support_dir = self.load_dataset_folder()

    def load_dataset_folder(self):
        meta = _load_meta(self.dataset_path)
        data_img = {}
        for class_name_one in sorted(meta.get('train', {}).keys()):
            if class_name_one != self.class_name:
                data_img[class_name_one] = _normal_images(self.dataset_path, class_name_one, 'train')
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():
            if len(data_img[class_name_one]) == 0:
                continue
            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []
                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while len(data_img[class_name_one]) > 1 and data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)
                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir


class FSAD_Dataset_test(Dataset):
    def __init__(
            self,
            dataset_path='../data/visa',
            class_name='candle',
            is_train=True,
            resize=256,
            shot=2,
    ):
        meta = _load_meta(dataset_path)
        class_pool = sorted(meta.get('test', {}).keys())
        assert class_name in class_pool, 'class_name: {}, should be in {}'.format(class_name, class_pool)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        self.query_dir, self.support_dir, self.query_mask = self.load_dataset_folder()
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), getattr(Image, 'Resampling', Image).LANCZOS),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), getattr(Image, 'Resampling', Image).NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        query_one, support_one, mask_one = self.query_dir[idx], self.support_dir[idx], self.query_mask[idx]
        query_img = Image.open(query_one).convert('RGB')
        query_img = self.transform_x(query_img)

        support_img = []
        for k in range(self.shot):
            support_img_one = Image.open(support_one[k]).convert('RGB')
            support_img_one = self.transform_x(support_img_one)
            support_img.append(support_img_one)

        if not mask_one:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = Image.open(mask_one).convert('L')
            mask = self.transform_mask(mask)
            y = 1
        return query_img, support_img, mask, y, query_one

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        meta = _load_meta(self.dataset_path)
        test_items = meta.get('test', {}).get(self.class_name, [])
        data_train = _normal_images(self.dataset_path, self.class_name, 'train')
        if len(data_train) == 0:
            raise RuntimeError(f'No normal training images found for class {self.class_name} in {self.dataset_path}')

        if self.shot > 0:
            torch.manual_seed(10)
            perm_indices = torch.randperm(len(data_train))
            indices = perm_indices[:min(self.shot, len(data_train))]
            data_train = [data_train[i] for i in indices]

        query_dir, support_dir, query_mask = [], [], []
        for item in test_items:
            support_dir_one = []
            query_dir.append(_resolve_path(self.dataset_path, item['img_path']))
            query_mask.append(
                _resolve_path(self.dataset_path, item.get('mask_path', ''))
                if int(item.get('anomaly', 0)) != 0 else None
            )
            for k in range(self.shot):
                random_choose = random.randint(0, (len(data_train) - 1))
                support_dir_one.append(data_train[random_choose])
            support_dir.append(support_dir_one)

        assert len(query_dir) == len(support_dir) == len(query_mask), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir, query_mask
