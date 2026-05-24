from .visa import FSAD_Dataset_train as _MetaDatasetTrain
from .visa import FSAD_Dataset_test as _MetaDatasetTest


class FSAD_Dataset_train(_MetaDatasetTrain):
    def __init__(self, dataset_path=r'C:\Users\Administrator\Desktop\dataset\MVTec_loco', class_name='breakfast_box', is_train=True, resize=256, shot=2, batch=32):
        super().__init__(dataset_path=dataset_path, class_name=class_name, is_train=is_train, resize=resize, shot=shot, batch=batch)


class FSAD_Dataset_test(_MetaDatasetTest):
    def __init__(self, dataset_path=r'C:\Users\Administrator\Desktop\dataset\MVTec_loco', class_name='breakfast_box', is_train=True, resize=256, shot=2):
        super().__init__(dataset_path=dataset_path, class_name=class_name, is_train=is_train, resize=resize, shot=shot)
