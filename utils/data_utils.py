import cv2
import math
import os
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely import wkt


CLASSES = {
    0: 'Road',
    1: 'Railway'
}


def get_all_data(data_dir, ground_truth, grid_sizes):
    """ Load all the training feature and label into memory. """

    all_train_names = list(ground_truth.ImageId.unique())

    train_ids_dict = dict(zip(np.arange(len(all_train_names)), all_train_names))

    img_ids = {39: train_ids_dict[39]}

    features_list = []
    labels_list = []

    # TODO Make generic
    # x_crop = 256 * 2
    # y_crop = 256 * 2

    for idx, img_key in img_ids.items():
        image_data = ImageData(data_dir, img_key, ground_truth, grid_sizes)
        image_data.create_train_feature()
        image_data.create_label()

        print(idx, image_data.train_feature.shape)
        print(idx, image_data.label.shape)

    #     features_list.append(image_data.train_feature[:x_crop, :y_crop, :])
    #     labels_list.append(image_data.label[:x_crop, :y_crop, :])
    #
    # features = np.stack(features_list, -1)
    # labels = np.stack(labels_list, -1)
    #
    # assert np.isfinite(features).all()
    # assert np.isfinite(labels).all() and (labels >= 0).all() and (labels <= 1).all()
    #
    # return np.rollaxis(features, 3, 0), np.rollaxis(labels, 3, 0)


class ImageData:

    def __init__(self, images_dir, image_key, ground_truth, grid_sizes, image_size=None):

        self.images_dir = images_dir
        self.image_key = image_key
        self.ground_truth = ground_truth
        self.grid_sizes = grid_sizes

        # TODO Make generic
        self.three_band_image = None
        self.ten_meters_band_image = None
        self.twenty_meters_band_image = None
        self.sixty_meters_band_image = None
        self.image = None
        self.image_size = image_size
        self.xy_min = None
        self.xy_max = None
        self.label = None
        self.train_feature = None

    # TODO Make generic
    def get_image_path(self):
        """ Returns the paths for all images. """
        return {
            '10': '{}/S2_10m/{}.tif'.format(self.images_dir, self.image_key),
            '20': '{}/S2_20m/{}.tif'.format(self.images_dir, self.image_key)
        }

    # TODO Make generic
    def read_image(self):
        """ Read all original images. """

        images = {}
        paths = self.get_image_path()

        for key in paths:
            im = tiff.imread(paths[key])
            images[key] = im

        if self.image_size:
            [height, width] = (self.image_size, self.image_size)

            im10 = images['10']
            im20 = images['20']

            images['10'] = cv2.resize(im10, (width, height), interpolation=cv2.INTER_CUBIC)
            images['20'] = cv2.resize(im20, (width, height), interpolation=cv2.INTER_CUBIC)

        return images

    # TODO Make generic
    def image_stack(self):
        """ Resample all images to highest resolution and align them. """

        images = self.read_image()

        im10 = images['10']
        im20 = images['20']

        # im20 = np.expand_dims(im20, axis=2)

        im = np.concatenate((im10, im20), axis=-1)

        return im

    # TODO Make generic
    def load_image(self):
        """
        Load three band and sixteen band images, registered and at the same resolution
        """

        self.image = self.image_stack()
        self.image_size = np.shape(self.image)[0:2]

        self.ten_meters_band_image = self.image[..., 0:4]
        self.twenty_meters_band_image = self.image[..., 4:6]

        # Retrieve image limits
        x_min, x_max, y_min, y_max = self.get_limits(self.image_key)
        self.xy_min = (x_min, y_min)
        self.xy_max = (x_max, y_max)

    def create_train_feature(self):
        """ Create synthesized features. """

        if self.image is None:
            self.load_image()

        img_10m = self.ten_meters_band_image[..., 0:].astype(np.float32)

        img_r = img_10m[..., 2:3]
        img_g = img_10m[..., 1:2]
        img_b = img_10m[..., 0:1]

        nir = img_10m[..., 3:4]

        # Compute Enhanced Vegetation Index (EVI)
        l, c1, c2 = 1.0, 6.0, 7.5
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = np.nan_to_num((nir - img_r) / (nir + c1 * img_r - c2 * img_b + l))
        evi = evi.clip(min=np.percentile(evi, 1), max=np.percentile(evi, 99))

        # Compute Normalized Difference Water Index (NDWI)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = np.nan_to_num((img_g - nir) / (img_g + nir))

        # Compute Soil-Adjusted Vegetation Index (SAVI)
        l_soil = 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            savi = np.nan_to_num((1 + l_soil) * (nir - img_r) / (nir + img_r + l_soil))

        # Concatenate features
        feature = np.concatenate([img_r, img_g, img_b, nir, evi, ndwi, savi], axis=2)

        assert not np.any(feature == np.inf)
        assert not np.any(feature == -np.inf)

        self.train_feature = feature

    def _get_scale_factors(self, img_size, xy_min, xy_max):
        """ Compute scaling factors. """
        x_max, y_max = xy_max
        x_min, y_min = xy_min
        h, w = img_size
        w_prime = w * w / (w + 0)
        h_prime = h * h / (h + 0)
        return w_prime / (x_max - x_min), h_prime / (y_max - y_min)

    def _convert_coordinates_to_raster(self, coordinates, img_size, xy_min, xy_max):
        """ Apply scaling factors to a given list of coordinates. """
        x_min, y_min = xy_min
        # Obtain scaling factors
        x_scale_factor, y_scale_factor = self._get_scale_factors(img_size, xy_min, xy_max)
        # Apply scaling factors
        coordinates[:, 0] -= x_min
        coordinates[:, 1] -= y_min
        coordinates[:, 0] *= x_scale_factor
        coordinates[:, 1] *= y_scale_factor
        coordinates[:, 1] *= -1
        coordinates[:, 1] += img_size[1]
        # Round and truncate scaled coordinates
        coordinates_scaled = np.round(coordinates).astype(np.int32)

        return coordinates_scaled

    def get_polygon_list(self, image_id, class_type):
        """
        Retrieve list of polygons (in the format of shapely polygon) loaded
        from WKT data (relative coordinates of polygons) from CSV file.
        """
        # Find polygon definitions for the given image id
        polygon_definitions = self.ground_truth[self.ground_truth.ImageId == image_id]
        # Find polygon definition for the given class type
        polygon_definitions = polygon_definitions[polygon_definitions.ClassType == class_type].MultipolygonWKT
        # Load WKT from multipolygon string definition
        polygon_list = None
        if len(polygon_definitions) > 0:
            assert len(polygon_definitions) == 1
            polygon_list = wkt.loads(polygon_definitions.values[0])

        return polygon_list

    def get_limits(self, image_id):
        limits = self.grid_sizes[self.grid_sizes.ImageId == image_id]

        x_min = 0.0 if math.isnan(limits.Xmin.values[0]) else limits.Xmin.values[0]
        x_max = limits.Xmax.values[0]
        y_min = limits.Ymin.values[0]
        y_max = 0.0 if math.isnan(limits.Ymax.values[0]) else limits.Ymax.values[0]

        return x_min, x_max, y_min, y_max

    def _get_and_convert_contours(self, polygon_list, raster_img_size, xy_min, xy_max):
        """ Convert parsed WKT polygons to contours as perimeter and interior lists of coordinates. """
        perimeter_list = []
        interior_list = []
        if polygon_list is None:
            return None
        for polygon_index in range(len(polygon_list)):
            polygon = polygon_list[polygon_index]
            perimeter = np.array(list(polygon.exterior.coords))
            perimeter_coords = self._convert_coordinates_to_raster(perimeter, raster_img_size, xy_min, xy_max)
            perimeter_list.append(perimeter_coords)
            for pi in polygon.interiors:
                interior = np.array(list(pi.coords))
                interior_coords = self._convert_coordinates_to_raster(interior, raster_img_size, xy_min, xy_max)
                interior_list.append(interior_coords)
        return perimeter_list, interior_list

    def _get_mask_from_contours(self, raster_img_size, contours, class_value=1):
        """ Convert contours to an image by filling perimeters as white and interiors as black. """
        img_mask = np.zeros(raster_img_size, np.uint8)
        if contours is None:
            return img_mask
        perimeter_list, interior_list = contours
        cv2.fillPoly(img_mask, perimeter_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask

    def create_label(self):
        """ Create the class labels. """
        if self.image is None:
            self.load_image()

        labels = np.zeros(np.append(self.image_size, len(CLASSES)), np.uint8)

        for class_id in CLASSES:
            # Get list of polygons
            polygon_list = self.get_polygon_list(self.image_key, class_id)
            # Convert obtained list of polygons to contours
            contours = self._get_and_convert_contours(polygon_list, self.image_size, self.xy_min, self.xy_max)
            # Convert contours to binary image
            mask = self._get_mask_from_contours(self.image_size, contours, 1)
            # Store binary image
            labels[..., class_id] = mask

        self.label = labels


def main():

    home_dir = os.environ['HOME']
    data_dir = f'{home_dir}/data/permafrost-imagery'

    _df_ground_truth = pd.read_csv(f'{data_dir}/train_wkt.csv',
                                   names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)

    _df_grid_sizes = pd.read_csv(f'{data_dir}/grid_sizes.csv',
                                 names=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'], skiprows=1)

    get_all_data(data_dir, _df_ground_truth, _df_grid_sizes)


if __name__ == '__main__':
    main()
