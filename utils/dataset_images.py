import argparse

import cv2
import os

import numpy as np
import pandas as pd
import progressbar

from codebase.utils.transforms import Normalize
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool
from skimage import io as skio
from torchvision import transforms
from utils.data_utils import ImageData

DATA_DIR = None
PHASE = None
THRESHOLD = 0.0
GROUND_TRUTH = None
GRID_SIZES = None
R_MIN = [0.1632990001142025, 0.11981919875741005, 0.12332499958574772, 0.11732500232756138, 0.06598571475063052, 0.04703333353002866, 0.038033333917458854, 0.03359999880194664, 0.03655000030994415, 0.07680000364780426, 0.12514999881386757, 0.1194990000873804]
G_MIN = [0.18540000170469284, 0.14661959940195085, 0.15399950288236142, 0.15985000133514404, 0.08517114345516477, 0.06486666699250539, 0.06299933309356372, 0.05790000036358833, 0.05950000137090683, 0.10090000182390213, 0.15155000239610672, 0.14764999598264694]
B_MIN = [0.2707490012049675, 0.1941799998283386, 0.18642500415444374, 0.18424999713897705, 0.10568571303571973, 0.08183333277702332, 0.07869933436314265, 0.07410000264644623, 0.08129999786615372, 0.14300000667572021, 0.21899999678134918, 0.22155000269412994]
R_MAX = [0.6658010053634642, 0.707700002193451, 0.7848259970545768, 0.7032500058412552, 0.23680057146719516, 0.1648013365268705, 0.15033399671316136, 0.1469999998807907, 0.14565099939703924, 0.5016999840736389, 0.6459509897232054, 0.6107500195503235]
G_MAX = [0.5540499985218048, 0.6205003924369812, 0.7105504953861236, 0.6515260055661201, 0.21835714365754808, 0.14756866127252546, 0.1402666668097178, 0.13680000603199005, 0.13585100471973405, 0.44710201323032345, 0.5557000041007996, 0.5141010153293608]
B_MAX = [0.6533009874820708, 0.6849804110527038, 0.7525249868631363, 0.6829250007867813, 0.2291717108232634, 0.14926733682552962, 0.1412340016166368, 0.13770000636577606, 0.1433500051498413, 0.48350200355052914, 0.6349000036716461, 0.6134509909152983]


def init_worker(data_dir=None, phase=None, threshold=None, ground_truth=None, grid_sizes=None):
    global DATA_DIR  # pylint: disable=global-statement
    global PHASE  # pylint: disable=global-statement
    global THRESHOLD  # pylint: disable=global-statement
    global GROUND_TRUTH  # pylint: disable=global-statement
    global GRID_SIZES  # pylint: disable=global-statement
    DATA_DIR = data_dir
    PHASE = phase
    THRESHOLD = threshold
    GROUND_TRUTH = ground_truth
    GRID_SIZES = grid_sizes


def _convert_one(img_key):

    images = OrderedDict()
    masks = OrderedDict()

    if os.path.exists(f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'):

        images[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'
        masks[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_mask.png'

        return images, masks

    date = datetime.strptime(img_key.split('_')[3], '%Y-%m-%d-%H-%M-%S')
    month = date.month - 1
    lower_percent = [R_MIN[month], G_MIN[month], B_MIN[month]]
    higher_percent = [R_MAX[month], G_MAX[month], B_MAX[month]]

    transform = transforms.Compose([Normalize(lower_percent=lower_percent,
                                              higher_percent=higher_percent,
                                              min_value=0.0,
                                              max_value=255.0)])

    image_data = ImageData(DATA_DIR, img_key, GROUND_TRUTH, GRID_SIZES, image_size=1024)
    image_data.create_train_feature()
    image_data.create_label()

    image = image_data.train_feature[..., [0, 1, 2]]
    label = image_data.label[..., 0]

    if label.sum() / 1024 / 1024 >= THRESHOLD:

        sample = {'image': image, 'label': label}
        sample = transform(sample)

        skio.imsave(f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg', sample['image'][:, :, (0, 1, 2)].astype(np.uint8), check_contrast=False)
        skio.imsave(f'{DATA_DIR}/{PHASE}/{img_key}_mask.png', 255 * sample['label'], check_contrast=False)

        images[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_sat.jpg'
        masks[img_key] = f'{DATA_DIR}/{PHASE}/{img_key}_mask.png'

        del sample

    del image_data

    return images, masks


def convert_all_data(data_dir, ground_truth, grid_sizes, phase, threshold, num_workers):

    image_ids = list(ground_truth.ImageId.unique())

    num_image_ids = len(image_ids)

    if not os.path.exists(f'{data_dir}/{phase}'):
        os.makedirs(f'{data_dir}/{phase}')

    images_rows = OrderedDict()
    masks_rows = OrderedDict()

    pool = Pool(initializer=init_worker, initargs=(data_dir, phase, threshold, ground_truth, grid_sizes), processes=num_workers)
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
    threshold = PARAMS.threshold

    _df_ground_truth = pd.read_csv(f'{data_dir}/{phase}_wkt.csv',
                                   names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)

    _df_grid_sizes = pd.read_csv(f'{data_dir}/grid_sizes.csv',
                                 names=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'], skiprows=1)

    convert_all_data(data_dir, _df_ground_truth, _df_grid_sizes, phase, threshold, num_workers)


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
        "--threshold",
        type=float,
        help="Positivity threshold",
        default=0.005
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel processes to launch",
        default=4
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
