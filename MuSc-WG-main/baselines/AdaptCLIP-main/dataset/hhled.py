"""HHLED dataset solver for anomaly detection."""

import json
from pathlib import Path


class HHLEDSolver:
    CLSNAMES = [
        'led_6core', 'led_8core',
    ]

    def __init__(self, root='../../data/hhled_AD'):
        self.root = Path(root)
        self.meta_path = self.root / 'meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            cls_dir = self.root / cls_name
            info['train'][cls_name] = []
            info['test'][cls_name] = []
            if not cls_dir.exists():
                continue

            for phase in ['train', 'test']:
                phase_dir = cls_dir / phase
                if not phase_dir.exists():
                    continue

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

                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1

                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

    @staticmethod
    def _mask_names(cls_dir, specie):
        mask_dir = cls_dir / 'ground_truth' / specie
        if not mask_dir.exists():
            return set()
        return {path.name for path in mask_dir.iterdir() if path.is_file()}


if __name__ == '__main__':
    runner = HHLEDSolver(root='../../data/hhled_AD')
    runner.run()
