import json
from pathlib import Path


class BTADSolver(object):
    def __init__(self, root=r'C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed'):
        self.root = Path(root)

    def run(self):
        return json.load(open(self.root / 'meta.json', 'r'))


if __name__ == '__main__':
    runner = BTADSolver()
    meta = runner.run()
    print({split: list(meta[split].keys()) for split in meta})
