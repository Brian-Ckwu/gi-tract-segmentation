import os
import math
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset

class GIImage(object):
    organs = ["stomach", "small_bowel", "large_bowel"]
    
    def __init__(self, fpath: Path, label_df: pd.DataFrame = None):
        case_day = str(fpath).split('/')[-3]
        fname = fpath.name # file name
        metadata = fname.rstrip(".png").split('_')
        slice_no = metadata[1]
        numbers = metadata[2:]
        # metadata: image id
        self.id = f"{case_day}_slice_{slice_no}"
        # metadata: slice width/height and pixel width/height
        self.sw = int(numbers[0])
        self.sh = int(numbers[1])
        self.pw = float(numbers[2])
        self.ph = float(numbers[3])
        # data: 2D array
        self.data = plt.imread(fpath)
        self.labels = None
        if label_df is not None:
            self.labels = self.get_labels(label_df)
    
    @property
    def tensor(self):
        return torch.from_numpy(self.data)
    
    def get_labels(self, label_df: pd.DataFrame) -> dict:
        labels = label_df.loc[label_df.id == self.id]
        organ2label = dict()
        for _, row in labels.iterrows():
            organ2label[row["class"]] = self.seg_to_label(row["segmentation"])
        return organ2label

    # converting run-length encoding to pixel-wise labels
    def seg_to_label(self, seg: str):
        label = np.zeros(shape=(self.sh, self.sw))
        if type(seg) == str:
            numbers = seg.split(' ')
            assert len(numbers) % 2 == 0
            for i in range(0, len(numbers), 2):
                start_id = int(numbers[i])
                length = int(numbers[i + 1])
                for j in range(length):
                    pixel = start_id + j
                    px = math.ceil(pixel / self.sw)
                    py = ((pixel - 1) % self.sw) + 1
                    label[px, py] = 1
        return label

    def label_to_seg(self, label):
        raise NotImplementedError
    
    def print_image_info(self) -> None:
        print(f"Image ID: {self.id}; slice width/height = ({self.sw}, {self.sh}); data shape = {self.data.shape}")
        
    def show_image(self) -> None:
        _, axs = plt.subplots(ncols=len(self.organs), squeeze=False, figsize=(15, 5))

        for i, organ in enumerate(GIImage.organs):
            axs[0, i].imshow(self.data, cmap="gray")
            if self.labels:
                axs[0, i].imshow(self.labels[organ], cmap="gray", alpha=0.4)
            axs[0, i].set_title(organ)

class GITractDataset(Dataset):
    
    def __init__(
        self,
        image_path: Path,
        label_path: Path = None,
        cases: set[str] = None
    ):
        if cases:
            self._image_paths = self.get_image_files_by_cases(image_path, cases)
        else:
            self._image_paths = [fpath for fpath in self.image_files_walker(image_path)]
        self._label_df = None
        if label_path:
            self._label_df = pd.read_csv(label_path)
    
    def __len__(self):
        return len(self._image_paths)
    
    def __getitem__(self, idx: int):
        return GIImage(fpath=self._image_paths[idx], label_df=self._label_df)
    
    @staticmethod
    def get_image_files_by_cases(image_path: str, cases: set[str]) -> list:
        image_path = Path(image_path)
        image_paths = []
        for case in os.listdir(image_path):
            if case in cases:
                case_path = image_path / case
                for day in os.listdir(case_path):
                    day_path = case_path / day / "scans"
                    for file in os.listdir(day_path):
                        fpath = day_path / file
                        image_paths.append(fpath)
        return image_paths

    @staticmethod
    def image_files_walker(image_path: str):
        for dirname, _, filenames in os.walk(image_path):
            dirpath = Path(dirname)
            for filename in filenames:
                yield dirpath / filename

def train_valid_split_cases(image_path: str, valid_size: float = 0.2) -> set:
    cases = os.listdir(image_path)
    valid_cases = set(random.sample(cases, math.ceil(len(cases) * valid_size)))
    train_cases = set(cases) - valid_cases
    assert ((valid_cases & train_cases) == set()) and ((valid_cases | train_cases) == set(cases))
    return train_cases, valid_cases
