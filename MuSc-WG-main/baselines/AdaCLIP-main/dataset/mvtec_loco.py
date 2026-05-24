import os
from .base_dataset import BaseDataset


MVTEC_LOCO_CLS_NAMES = [
    'breakfast_box',
    'juice_bottle',
    'pushpins',
    'screw_bag',
    'splicing_connectors',
]
MVTEC_LOCO_ROOT = r'C:\Users\Administrator\Desktop\dataset\MVTec_loco'


class MVTecLOCODataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MVTEC_LOCO_CLS_NAMES, aug_rate=0.0, root=MVTEC_LOCO_ROOT, training=True):
        super(MVTecLOCODataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
