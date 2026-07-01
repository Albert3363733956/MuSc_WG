import json
from pathlib import Path

from dataset import MICROLED_ROOT


class MicroLEDSolver(object):
    CLSNAMES = [
        'microled_TypeA_1', 'microled_TypeA_2', 'microled_TypeA_3', 'microled_TypeA_4',
        'microled_TypeA_5', 'microled_TypeA_6', 'microled_TypeA_7', 'microled_TypeA_8',
        'microled_TypeA_9', 'microled_TypeA_10', 'microled_TypeA_11', 'microled_TypeA_12',
        'microled_TypeA_13', 'microled_TypeA_14', 'microled_TypeA_15', 'microled_TypeA_16',
        'microled_TypeA_17', 'microled_TypeA_18', 'microled_TypeA_19', 'microled_TypeA_20',
    ]

    def __init__(self, root=MICROLED_ROOT):
        self.root = Path(root)
        self.meta_path = self.root / 'meta.json'

    def run(self):
        self.generate_meta_info()

    def generate_meta_info(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = self.root / cls_name
            for phase in ['train', 'test']:
                info[phase][cls_name] = self._collect_phase(cls_name, cls_dir, phase)

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def _collect_phase(self, cls_name, cls_dir, phase):
        phase_dir = cls_dir / phase
        if not phase_dir.exists():
            return []

        cls_info = []
        for specie_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            specie = specie_dir.name
            is_abnormal = specie != 'good'
            img_names = sorted(path.name for path in specie_dir.iterdir() if path.is_file())
            mask_names = self._mask_names(cls_dir, specie) if phase == 'test' and is_abnormal else set()

            for img_name in img_names:
                mask_path = ''
                if phase == 'test' and is_abnormal:
                    if img_name not in mask_names:
                        raise FileNotFoundError(
                            f'Missing mask for image: {cls_dir / phase / specie / img_name}'
                        )
                    mask_path = f'{cls_name}/ground_truth/{specie}/{img_name}'

                cls_info.append(dict(
                    img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                    mask_path=mask_path,
                    cls_name=cls_name,
                    specie_name=specie,
                    anomaly=1 if is_abnormal else 0,
                ))

        return cls_info

    @staticmethod
    def _mask_names(cls_dir, specie):
        mask_dir = cls_dir / 'ground_truth' / specie
        if not mask_dir.exists():
            return set()
        return {path.name for path in mask_dir.iterdir() if path.is_file()}


if __name__ == '__main__':
    runner = MicroLEDSolver()
    runner.run()
