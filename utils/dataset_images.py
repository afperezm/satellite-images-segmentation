import argparse

import cv2
import os
import numpy as np
import pandas as pd
import progressbar

from codebase.utils.transforms import Normalize
from torchvision import transforms
from utils.data_utils import ImageData


def convert_all_data(data_dir, ground_truth, grid_sizes, phase):

    all_train_names = list(ground_truth.ImageId.unique())

    img_ids = dict(zip(np.arange(len(all_train_names)), all_train_names))

    num_images = len(all_train_names)

    transform = transforms.Compose([Normalize(min_value=0.0, max_value=255.0, lower_percent=1, higher_percent=99)])

    if not os.path.exists(f'{data_dir}/{phase}'):
        os.makedirs(f'{data_dir}/{phase}')

    bar = progressbar.ProgressBar(max_value=num_images)
    for idx, img_key in img_ids.items():

        if os.path.exists(f'{data_dir}/{phase}/{img_key}_sat.jpg'):
            continue

        image_data = ImageData(data_dir, img_key, ground_truth, grid_sizes, image_size=1024)
        image_data.create_train_feature()
        image_data.create_label()

        image = image_data.train_feature
        label = image_data.label[..., 0]

        if label.sum() / 1024 / 1024 >= 0.005:

            sample = {'image': image, 'label': label}
            sample = transform(sample)

            cv2.imwrite(f'{data_dir}/{phase}/{img_key}_sat.jpg', sample['image'][:, :, (0, 1, 2)])
            cv2.imwrite(f'{data_dir}/{phase}/{img_key}_mask.png', 255 * sample['label'])

            del sample

        del image_data

        bar.update(idx)

    bar.update(num_images)


def main():

    data_dir = PARAMS.data_dir
    phase = PARAMS.phase

    _df_ground_truth = pd.read_csv(f'{data_dir}/{phase}_wkt.csv',
                                   names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)

    _df_grid_sizes = pd.read_csv(f'{data_dir}/grid_sizes.csv',
                                 names=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'], skiprows=1)

    convert_all_data(data_dir, _df_ground_truth, _df_grid_sizes, phase)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to convert dataset TIFF images to PNG and masks")
    parser.add_argument(
        "--data_dir",
        help="Dataset directory",
        required=True
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["train", "test"]
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
