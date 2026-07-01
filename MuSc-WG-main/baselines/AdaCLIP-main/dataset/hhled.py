import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

HHLED_CLS_NAMES = [
    'led_6core', 'led_8core',
]
HHLED_ROOT = os.environ.get('ADACLIP_HHLED_ROOT', os.path.join(DATA_ROOT, 'hhled_AD'))

class HHLEDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=HHLED_CLS_NAMES, aug_rate=0.2, root=HHLED_ROOT, training=True):
        super(HHLEDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
