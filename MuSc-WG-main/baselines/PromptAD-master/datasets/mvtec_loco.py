import json
import os
import random


mvtec_loco_classes = [
    'breakfast_box',
    'juice_bottle',
    'pushpins',
    'screw_bag',
    'splicing_connectors',
]
MVTEC_LOCO_DIR = r'C:\Users\Administrator\Desktop\dataset\MVTec_loco'


def _load_meta_split(data_path, category, split):
    with open(os.path.join(data_path, 'meta.json'), 'r') as f:
        meta = json.load(f)
    return meta[split][category]


def _items_to_tuple(items, data_path):
    img_paths, gt_paths, labels, types = [], [], [], []
    for item in items:
        img_paths.append(os.path.join(data_path, item['img_path']))
        labels.append(int(item['anomaly']))
        types.append(item.get('specie_name', 'good'))
        if int(item['anomaly']) == 0:
            gt_paths.append(0)
        else:
            gt_paths.append(os.path.join(data_path, item['mask_path']))
    return img_paths, gt_paths, labels, types


def load_mvtec_loco(category, k_shot, data_path=None):
    if data_path is None:
        data_path = MVTEC_LOCO_DIR
    assert category in mvtec_loco_classes

    train_items = _load_meta_split(data_path, category, 'train')
    test_items = _load_meta_split(data_path, category, 'test')

    if k_shot > 0 and len(train_items) > k_shot:
        train_items = random.Random(10).sample(train_items, k_shot)

    return _items_to_tuple(train_items, data_path), _items_to_tuple(test_items, data_path)
