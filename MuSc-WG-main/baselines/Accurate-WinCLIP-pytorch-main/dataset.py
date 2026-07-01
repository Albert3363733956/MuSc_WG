import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

Vis_CLSNAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

Vis_CLSNAMES_map_index = {}
for k, index in zip(Vis_CLSNAMES, range(len(Vis_CLSNAMES))):
	Vis_CLSNAMES_map_index[k] = index

CLSNAMES = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
CLSNAMES_map_index = {}
for k, index in zip(CLSNAMES, range(len(CLSNAMES))):
	CLSNAMES_map_index[k] = index

MINILED_CLSNAMES = ['miniled_TypeA_1', 'miniled_TypeA_2', 'miniled_TypeB_1', 'miniled_TypeB_2']
MICROLED_CLSNAMES = [
	'microled_TypeA_1', 'microled_TypeA_2', 'microled_TypeA_3', 'microled_TypeA_4',
	'microled_TypeA_5', 'microled_TypeA_6', 'microled_TypeA_7', 'microled_TypeA_8',
	'microled_TypeA_9', 'microled_TypeA_10', 'microled_TypeA_11', 'microled_TypeA_12',
	'microled_TypeA_13', 'microled_TypeA_14', 'microled_TypeA_15', 'microled_TypeA_16',
	'microled_TypeA_17', 'microled_TypeA_18', 'microled_TypeA_19', 'microled_TypeA_20'
]
HHLED_CLSNAMES = ['led_6core', 'led_8core']

DATASET_CLSNAMES = {
	'mvtec': CLSNAMES,
	'visa': Vis_CLSNAMES,
	'miniled': MINILED_CLSNAMES,
	'microled': MICROLED_CLSNAMES,
	'hhled': HHLED_CLSNAMES,
}

IMAGE_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')


def normalize_dataset_name(dataset_name):
	dataset_key = str(dataset_name).lower()
	aliases = {
		'mvtec_ad': 'mvtec',
		'miniled_ad': 'miniled',
		'microled_ad': 'microled',
		'hhled_ad': 'hhled',
	}
	return aliases.get(dataset_key, dataset_key)


def _class_index_map(class_names):
	return {name: index for index, name in enumerate(class_names)}


def _meta_path(root):
	return os.path.join(root, 'meta.json')


def has_meta(root):
	return os.path.isfile(_meta_path(root))


def _load_meta(root):
	with open(_meta_path(root), 'r') as f:
		return json.load(f)


def get_dataset_classnames(dataset_name, root=None, mode='test'):
	dataset_key = normalize_dataset_name(dataset_name)
	if root is not None and has_meta(root):
		meta_info = _load_meta(root)
		if mode in meta_info and len(meta_info[mode]) > 0:
			return list(meta_info[mode].keys())

	default_names = DATASET_CLSNAMES.get(dataset_key, [])
	if root is not None and default_names:
		existing_names = [name for name in default_names if os.path.isdir(os.path.join(root, name))]
		if existing_names:
			return existing_names

	if default_names:
		return list(default_names)

	if root is not None and os.path.isdir(root):
		return sorted(
			name for name in os.listdir(root)
			if os.path.isdir(os.path.join(root, name, mode))
		)

	return []


def _select_classnames(available_names, obj_name):
	if obj_name is not None and obj_name != 'all':
		return obj_name if isinstance(obj_name, list) else [obj_name]
	return list(available_names)



class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name is not None and obj_name != "all":
				if isinstance(obj_name, list):
					self.cls_names = obj_name
				else:
					self.cls_names = [obj_name]
			else:
				self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				# Use fixed seed and sampling without replacement (like random.sample)
				torch.manual_seed(10)
				# Generate random permutation of indices
				perm_indices = torch.randperm(len(data_tmp))
				# Take the first k_shot indices
				indices = perm_indices[:min(k_shot, len(data_tmp))]
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), "cls_id": Vis_CLSNAMES_map_index.get(cls_name, 0)}



class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None, class_names=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]
		if class_names is None:
			class_names = list(meta_info.keys()) if obj_name == 'all' or obj_name is None else _select_classnames(meta_info.keys(), obj_name)
		self.class_index_map = _class_index_map(class_names)

		if mode == 'train':
			if isinstance(obj_name, list):
				self.cls_names = obj_name
			else:
				self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name is not None and obj_name != "all":
				if isinstance(obj_name, list):
					self.cls_names = obj_name
				else:
					self.cls_names = [obj_name]
			else:
				self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				# Use fixed seed and sampling without replacement (like random.sample)
				torch.manual_seed(10)
				# Generate random permutation of indices
				perm_indices = torch.randperm(len(data_tmp))
				# Take the first k_shot indices
				indices = perm_indices[:min(k_shot, len(data_tmp))]
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length


	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']

		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), "cls_id": self.class_index_map.get(cls_name, 0)}


class FolderAnomalyDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', k_shot=0, save_dir=None,
				 obj_name=None, class_names=None, dataset_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.mode = mode
		self.dataset_name = normalize_dataset_name(dataset_name or '')
		available_names = class_names or get_dataset_classnames(self.dataset_name, root=root, mode=mode)
		self.cls_names = _select_classnames(available_names, obj_name)
		self.class_index_map = _class_index_map(available_names)

		if mode == 'train':
			save_dir = os.path.join(save_dir, 'k_shot.txt') if save_dir is not None else None

		self.data_all = self._scan()
		if mode == 'train' and k_shot > 0:
			torch.manual_seed(10)
			perm_indices = torch.randperm(len(self.data_all))
			indices = perm_indices[:min(k_shot, len(self.data_all))]
			self.data_all = [self.data_all[i] for i in indices]
			if save_dir is not None:
				with open(save_dir, 'a') as f:
					for data_tmp in self.data_all:
						f.write(data_tmp['img_path'] + '\n')
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	@staticmethod
	def _list_dirs(path):
		if not os.path.isdir(path):
			return []
		return sorted(
			name for name in os.listdir(path)
			if os.path.isdir(os.path.join(path, name))
		)

	@staticmethod
	def _list_files(path):
		if not os.path.isdir(path):
			return []
		return sorted(
			name for name in os.listdir(path)
			if os.path.isfile(os.path.join(path, name)) and name.lower().endswith(IMAGE_EXTENSIONS)
		)

	def _find_mask_name(self, cls_name, specie_name, img_name, img_index):
		mask_dir = os.path.join(self.root, cls_name, 'ground_truth', specie_name)
		mask_names = self._list_files(mask_dir)
		if img_name in mask_names:
			return img_name
		if img_index < len(mask_names):
			return mask_names[img_index]
		raise FileNotFoundError(f'Missing mask for image: {os.path.join(self.root, cls_name, self.mode, specie_name, img_name)}')

	def _scan(self):
		data_all = []
		for cls_name in self.cls_names:
			phase_dir = os.path.join(self.root, cls_name, self.mode)
			specie_names = self._list_dirs(phase_dir)
			if self.mode == 'train' and 'good' in specie_names:
				specie_names = ['good']
			for specie_name in specie_names:
				is_abnormal = specie_name != 'good'
				img_names = self._list_files(os.path.join(phase_dir, specie_name))
				for img_index, img_name in enumerate(img_names):
					mask_path = ''
					if self.mode == 'test' and is_abnormal:
						mask_name = self._find_mask_name(cls_name, specie_name, img_name, img_index)
						mask_path = os.path.join(cls_name, 'ground_truth', specie_name, mask_name)
					data_all.append(dict(
						img_path=os.path.join(cls_name, self.mode, specie_name, img_name),
						mask_path=mask_path,
						cls_name=cls_name,
						specie_name=specie_name,
						anomaly=1 if is_abnormal else 0,
					))
		return data_all

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']

		img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), 'cls_id': self.class_index_map.get(cls_name, 0)}

