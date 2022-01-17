import argparse
import csv
from collections import OrderedDict

import gdal
import geopandas as gpd
import glob
import os
import subprocess

from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from sklearn.model_selection import train_test_split

ROADS_BUFFERED = None
OUTPUT_DIR = None
CLASS_IDX = None


def _compose_one(cell_idx, polygon_str, image_filename):

    grid_sizes_rows = OrderedDict()
    train_polys_rows = OrderedDict()

    band_name = os.path.basename(os.path.dirname(image_filename))

    if band_name == 'S2_20m':
        return grid_sizes_rows, train_polys_rows

    # Load polygons in WKT format from string
    polygon_wkt = wkt.loads(polygon_str)

    # Retrieve tile corners
    top_left = (polygon_wkt.exterior.xy[0][3], polygon_wkt.exterior.xy[1][3])
    top_right = (polygon_wkt.exterior.xy[0][2], polygon_wkt.exterior.xy[1][2])
    bottom_right = (polygon_wkt.exterior.xy[0][1], polygon_wkt.exterior.xy[1][1])
    bottom_left = (polygon_wkt.exterior.xy[0][0], polygon_wkt.exterior.xy[1][0])

    # Compute limits
    x_min = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    x_max = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    y_min = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    y_max = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])

    # Compose shapely polygon for the current grid tile
    grid_polygon = Polygon([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])

    image = gdal.Open(image_filename)

    image_gt = image.GetGeoTransform()
    image_width = image.RasterXSize
    image_height = image.RasterYSize

    xmin = image_gt[0]
    xmax = image_gt[0] + image_width * image_gt[1]
    ymin = image_gt[3] + image_height * image_gt[5]
    ymax = image_gt[3]

    image_polygon = Polygon([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

    if grid_polygon.intersection(image_polygon).area == 0.0:
        return grid_sizes_rows, train_polys_rows

    # Clip roads by grid polygon
    roads_clipped = gpd.clip(ROADS_BUFFERED, grid_polygon)

    if roads_clipped.empty:
        roads_clipped_wkt = MultiPolygon([roads_clipped.any()]).wkt
    elif roads_clipped.geom_type[0] == 'Polygon':
        roads_clipped_wkt = MultiPolygon([roads_clipped.all()]).wkt
    elif roads_clipped.geom_type[0] == 'MultiPolygon':
        roads_clipped_wkt = MultiPolygon(roads_clipped.all()).wkt
    else:
        raise ValueError('Unknown geometry type')

    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
    out_image_name = f'{image_basename}-{cell_idx:04d}'

    if not os.path.exists(os.path.join(OUTPUT_DIR, band_name)):
        os.makedirs(os.path.join(OUTPUT_DIR, band_name))

    out_filename = os.path.join(OUTPUT_DIR, band_name, f'{out_image_name}.tif')

    print(f'- Clipping image {image_basename} around grid polygon {cell_idx:04d}')

    # Clip image around grid polygon
    if not os.path.exists(out_filename):
        subprocess.check_call(['gdalwarp', '-te', str(x_min), str(y_min), str(x_max), str(y_max),
                               image_filename, out_filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print('  Done')

    if f'{out_image_name}_{CLASS_IDX}' not in grid_sizes_rows:
        grid_sizes_rows[f'{out_image_name}_{CLASS_IDX}'] = (out_image_name, x_min, x_max, y_min, y_max)

    if f'{out_image_name}_{CLASS_IDX}' not in train_polys_rows:
        train_polys_rows[f'{out_image_name}_{CLASS_IDX}'] = (out_image_name, CLASS_IDX, roads_clipped_wkt)

    return grid_sizes_rows, train_polys_rows


def _load_roads(roads_shp, roads_crs):

    print("- Loading roads shapefile")

    roads = gpd.read_file(roads_shp)
    roads = roads.to_crs(roads_crs)

    roads_selected = roads[['geometry']]

    print(f"  Done, got {len(roads)} features of type {roads_selected.geom_type[0]}")

    print("- Dissolving roads")

    roads_dissolved = roads_selected.dissolve()

    print(f"  Done, got {len(roads_dissolved)} features of type {roads_dissolved.geom_type[0]}")

    print("- Creating buffer around roads")

    roads_buffered = roads_dissolved.buffer(5)

    print(f"  Done, got {len(roads_buffered)} features of type {roads_buffered.geom_type[0]}")

    return roads_buffered


def _compose_dataset(output_dir, grid_csv, roads_shp, epsg_crs, class_idx):

    global ROADS_BUFFERED
    global OUTPUT_DIR
    global CLASS_IDX

    OUTPUT_DIR = output_dir
    CLASS_IDX = class_idx

    images_dir = os.path.dirname(grid_csv)

    grid_sizes_rows = OrderedDict()
    ground_truth_rows = OrderedDict()

    # Load roads shape
    ROADS_BUFFERED = _load_roads(roads_shp, epsg_crs)

    # Lookup list of images
    images = sorted(glob.glob(os.path.join(images_dir, '**', f'*_eopatch-*.tif')))

    # Build list of polygons
    polygons = []

    with open(grid_csv, "rt", encoding="utf-8", newline="") as grid_csv_file:
        # Init CSV reader
        reader = csv.DictReader(grid_csv_file, fieldnames=['Id', 'PolygonWkt'])

        # Skip first CSV row
        next(reader)

        # Loop over all CSV rows
        for row_idx, row in enumerate(reader):

            # Retrieve polygon string in WKT format
            polygon_str = row['PolygonWkt']

            # Append polygon string to list
            polygons.append((row_idx, polygon_str))

    for polygon, image in [(p, i) for p in polygons for i in images]:
        grid_size_row, ground_truth_row = _compose_one(polygon[0],
                                                       polygon[1],
                                                       image)
        grid_sizes_rows.update(grid_size_row)
        ground_truth_rows.update(ground_truth_row)

    return grid_sizes_rows, ground_truth_rows


def _save_dataset(output_dir, grid_sizes, all_wkt_polygons):

    # train_wkt_polygons, test_wkt_polygons = train_test_split(list(all_wkt_polygons.values()), random_state=42)

    def parse_key(key):
        print(key)
        return int(key.split('_')[-1].split('-')[1])

    num_patches = len(set([parse_key(value[0]) for value in all_wkt_polygons.values()]))

    patches_indices = [idx for idx in range(num_patches)]
    patches_train_indices, patches_test_indices = train_test_split(patches_indices, random_state=42)

    train_wkt_polygons = [value for value in all_wkt_polygons.values() if
                          parse_key(value[0]) in patches_train_indices]
    test_wkt_polygons = [value for value in all_wkt_polygons.values() if
                         parse_key(value[0]) in patches_test_indices]

    grid_sizes_csv = os.path.join(output_dir, 'grid_sizes.csv')
    print(f"- Saving grid sizes in CSV format to: {grid_sizes_csv}")
    with open(grid_sizes_csv, "wt", encoding="utf-8", newline="") as grid_sizes_csv_file:
        writer = csv.DictWriter(grid_sizes_csv_file, fieldnames=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'])
        writer.writeheader()
        for image_id, x_min, x_max, y_min, y_max in list(grid_sizes.values()):
            writer.writerow({'ImageId': image_id, 'Xmax': x_max, 'Ymin': y_min, 'Xmin': x_min, 'Ymax': y_max})
    print("  Done")

    train_wkt_csv = os.path.join(output_dir, 'train_wkt.csv')
    print(f"- Saving WKT train polygons in CSV format to: {train_wkt_csv}")
    with open(train_wkt_csv, "wt", encoding="utf-8", newline="") as train_wkt_csv_file:
        writer = csv.DictWriter(train_wkt_csv_file, fieldnames=['ImageId', 'ClassType', 'MultipolygonWKT'])
        writer.writeheader()
        for image_id, class_type, multipolygon_wkt in train_wkt_polygons:
            writer.writerow({'ImageId': image_id, 'ClassType': class_type, 'MultipolygonWKT': multipolygon_wkt})
    print("  Done")

    test_wkt_csv = os.path.join(output_dir, 'test_wkt.csv')
    print(f"- Saving WKT test polygons in CSV format to: {test_wkt_csv}")
    with open(test_wkt_csv, "wt", encoding="utf-8", newline="") as test_wkt_csv_file:
        writer = csv.DictWriter(test_wkt_csv_file, fieldnames=['ImageId', 'ClassType', 'MultipolygonWKT'])
        writer.writeheader()
        for image_id, class_type, multipolygon_wkt in test_wkt_polygons:
            writer.writerow({'ImageId': image_id, 'ClassType': class_type, 'MultipolygonWKT': multipolygon_wkt})
    print("  Done")


def main():
    output_dir = PARAMS.output_dir
    grids_files = PARAMS.grids_files
    roads_files = PARAMS.roads_files
    crs_list = PARAMS.crs_list

    assert len(grids_files) == len(roads_files) == len(crs_list)

    grid_sizes, ground_truth_polygons = OrderedDict(), OrderedDict()

    for idx in range(0, len(grids_files)):
        grid_sizes_rows, ground_truth_rows = _compose_dataset(output_dir,
                                                              grids_files[idx],
                                                              roads_files[idx],
                                                              crs_list[idx],
                                                              idx)

        grid_sizes.update(grid_sizes_rows)
        ground_truth_polygons.update(ground_truth_rows)

    _save_dataset(output_dir, grid_sizes, ground_truth_polygons)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to download patches from a grid of polygons in WKT format")
    parser.add_argument(
        "--output_dir",
        help="Output directory where to store generated dataset files",
        required=True
    )
    parser.add_argument(
        "--grids_files",
        nargs='+',
        help="Grids CSV files (in WKT format) with polygons (coordinates in EPSG 4326) of the areas downloaded",
        required=True
    )
    parser.add_argument(
        "--roads_files",
        nargs='+',
        help="Roads shape files",
        required=True
    )
    parser.add_argument(
        "--crs_list",
        nargs='+',
        help="Output coordinate reference systems to use",
        default="epsg:3978"
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
