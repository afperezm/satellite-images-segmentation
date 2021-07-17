import argparse
import csv
import glob
import os
import subprocess

import gdal
import osr
from pyproj import Transformer
from shapely import wkt

FIELDNAMES = ['Id', 'PolygonWkt']
PARAMS = None


def _compute_limits(grid_csv, crs='epsg:32614'):

    # Set up transformers
    to_planar_transformer = Transformer.from_crs('epsg:4326', crs, always_xy=True)

    # Initialize limits
    x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')

    with open(grid_csv, "rt", encoding="utf-8", newline="") as grid_csv_file:
        # Init CSV reader
        reader = csv.DictReader(grid_csv_file, fieldnames=FIELDNAMES)

        # Skip first CSV row
        next(reader)

        # Loop over all CSV rows
        for csv_idx, csv_row in enumerate(reader):
            # Retrieve polygon string in WKT format
            polygon_str = csv_row[FIELDNAMES[1]]

            # Load polygons in WKT format from string
            polygon_wkt = wkt.loads(polygon_str)

            # Retrieve tile corners
            top_left = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][3], polygon_wkt.exterior.xy[1][3])
            top_right = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][2], polygon_wkt.exterior.xy[1][2])
            bottom_right = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][1], polygon_wkt.exterior.xy[1][1])
            bottom_left = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][0], polygon_wkt.exterior.xy[1][0])

            # Compute limits
            x_min = min(x_min, top_left[0], bottom_left[0], top_right[0], bottom_right[0])
            x_max = max(x_max, top_left[0], bottom_left[0], top_right[0], bottom_right[0])
            y_min = min(y_min, top_left[1], bottom_left[1], top_right[1], bottom_right[1])
            y_max = max(y_max, top_left[1], bottom_left[1], top_right[1], bottom_right[1])

    return x_min, x_max, y_min, y_max


def _retrieve_images(grid_csv, satellites_list, resolution_list):
    data_dir = os.path.dirname(grid_csv)

    # Retrieve GeoTiff images
    images = glob.glob('{}/*/*/*/*.tif'.format(data_dir))  # grid / satellite / resolution / name

    # Retrieve coordinate reference system (CRS)
    geotiff_image = gdal.Open(images[0])

    geotiff_projection = osr.SpatialReference(wkt=geotiff_image.GetProjection())
    geotiff_projection.AutoIdentifyEPSG()

    geotiff_crs = geotiff_projection.GetAttrValue('AUTHORITY', 1)

    # Recompose list of GeoTiff images as a dictionary
    images_dict = {}

    for image in images:
        basename = os.path.splitext(os.path.basename(image))[0]
        basename_split = basename.split('_')
        # Pop grid index
        basename_split.pop(2)
        # Retrieve satellite
        satellite = basename_split[1]
        if satellite not in satellites_list:
            continue
        # Retrieve resolution
        resolution = basename_split[2]
        if resolution not in resolution_list:
            continue
        key = '_'.join(basename_split)
        if key in images_dict:
            images_dict[key].append(image)
        else:
            images_dict[key] = [image]

    return images_dict, f"epsg:{geotiff_crs}"


def _merge_images(grid_csv, images_dict, upper_left, lower_right):
    data_dir = os.path.dirname(grid_csv)
    out_dir = os.path.join(data_dir, 'images')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for key in images_dict:
        out_filename = '{}/{}.tif'.format(out_dir, key)
        print('Processing {}'.format(key))
        if os.path.exists(out_filename):
            print('Skipping, already processed')
        else:
            subprocess.check_call(
                ['gdal_merge.py', '-o', out_filename, '-ul_lr', str(upper_left[0]), str(upper_left[1]),
                 str(lower_right[0]), str(lower_right[1])] + images_dict[key], stderr=subprocess.STDOUT)


def main():
    grid_file = PARAMS.grid_file
    satellites_list = PARAMS.satellites_list
    resolution_list = PARAMS.resolution_list

    images_dict, crs = _retrieve_images(grid_file, satellites_list, resolution_list)

    x_min, x_max, y_min, y_max = _compute_limits(grid_file, crs)

    _merge_images(grid_file, images_dict, (x_min, y_max), (x_max, y_min))


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to merge grid images")
    parser.add_argument(
        "--grid_file",
        help="Grid file with polygons of the downloaded areas",
        required=True
    )
    parser.add_argument(
        "--satellites_list",
        nargs="+",
        default=["S2"]
    )
    parser.add_argument(
        "--resolution_list",
        nargs="+",
        default=["10m"]
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
