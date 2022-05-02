import argparse

import cv2
import os
import pandas as pd
import progressbar

from codebase.utils.transforms import Normalize
from collections import OrderedDict
from multiprocessing import Pool
from torchvision import transforms
from utils.data_utils import ImageData

DATA_DIR = None
PHASE = None
GROUND_TRUTH = None
GRID_SIZES = None


def init_worker(data_dir=None, phase=None, ground_truth=None, grid_sizes=None):
    global DATA_DIR  # pylint: disable=global-statement
    global PHASE  # pylint: disable=global-statement
    global GROUND_TRUTH  # pylint: disable=global-statement
    global GRID_SIZES  # pylint: disable=global-statement
    DATA_DIR = data_dir
    PHASE = phase
    GROUND_TRUTH = ground_truth
    GRID_SIZES = grid_sizes


def _convert_one(img_key):

    images = OrderedDict()
    masks = OrderedDict()

    if os.path.exists(f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'):

        images[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'
        masks[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_mask.png'

        return images, masks

    transform = transforms.Compose([Normalize(min_value=0.0, max_value=255.0, lower_percent=1, higher_percent=99)])

    image_data = ImageData(DATA_DIR, img_key, GROUND_TRUTH, GRID_SIZES, image_size=1024)
    image_data.create_train_feature()
    image_data.create_label()

    image = image_data.train_feature
    label = image_data.label[..., 0]

    if label.sum() / 1024 / 1024 >= 0.005:

        sample = {'image': image, 'label': label}
        sample = transform(sample)

        cv2.imwrite(f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg', sample['image'][:, :, (0, 1, 2)])
        cv2.imwrite(f'{DATA_DIR}/{PHASE}/{img_key}_mask.png', 255 * sample['label'])

        images[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'
        masks[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_mask.png'

        del sample

    del image_data

    return images, masks


def convert_all_data(data_dir, ground_truth, grid_sizes, phase, num_workers):

    image_ids = list(ground_truth.ImageId.unique())

    num_image_ids = len(image_ids)

    if not os.path.exists(f'{data_dir}/{phase}'):
        os.makedirs(f'{data_dir}/{phase}')

    images_rows = OrderedDict()
    masks_rows = OrderedDict()

    pool = Pool(initializer=init_worker, initargs=(data_dir, phase, ground_truth, grid_sizes), processes=num_workers)
    bar = progressbar.ProgressBar(max_value=num_image_ids)
    for idx, processed in enumerate(pool.imap_unordered(_convert_one, image_ids), start=1):
        image_row = processed[0]
        mask_row = processed[1]
        images_rows.update(image_row)
        masks_rows.update(mask_row)
        bar.update(idx)
    bar.update(num_image_ids)
    pool.close()
    pool.join()

    assert len(images_rows) == len(masks_rows)

    print(f'Processed successfully {len(images_rows)} images')


def main():

    data_dir = PARAMS.data_dir
    phase = PARAMS.phase
    num_workers = PARAMS.num_workers

    _df_ground_truth = pd.read_csv(f'{data_dir}/{phase}_wkt.csv',
                                   names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)

    _df_grid_sizes = pd.read_csv(f'{data_dir}/grid_sizes.csv',
                                 names=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'], skiprows=1)

    convert_all_data(data_dir, _df_ground_truth, _df_grid_sizes, phase, num_workers)


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
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel processes to launch",
        default=4)
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
