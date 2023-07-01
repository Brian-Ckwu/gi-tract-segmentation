import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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