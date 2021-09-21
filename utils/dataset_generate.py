import argparse
import csv
from collections import OrderedDict

import geopandas as gpd
import glob
import os
import subprocess

from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon


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

    roads_buffered = roads_dissolved.buffer(10)

    print(f"  Done, got {len(roads_buffered)} features of type {roads_buffered.geom_type[0]}")

    return roads_buffered


def _compose_dataset(grid_csv, roads_shp, epsg_crs):

    images_dir = os.path.dirname(grid_csv)
    output_dir = os.path.dirname(images_dir)

    grid_sizes_rows = OrderedDict()
    train_wkt_rows = OrderedDict()

    # Load roads shape
    roads_buffered = _load_roads(roads_shp, epsg_crs)

    # Set up transformers
    to_planar_transformer = Transformer.from_crs('epsg:4326', epsg_crs, always_xy=True)

    with open(grid_csv, "rt", encoding="utf-8", newline="") as grid_csv_file:
        # Init CSV reader
        reader = csv.DictReader(grid_csv_file, fieldnames=['Id', 'PolygonWkt'])

        # Skip first CSV row
        next(reader)

        # Loop over all CSV rows
        for idx, row in enumerate(reader):

            print(f'- Loading grid cell {idx}')

            # Retrieve polygon string in WKT format
            polygon_str = row['PolygonWkt']

            # Load polygons in WKT format from string
            polygon_wkt = wkt.loads(polygon_str)

            print('  Done')

            # Retrieve tile corners
            top_left = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][3], polygon_wkt.exterior.xy[1][3])
            top_right = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][2], polygon_wkt.exterior.xy[1][2])
            bottom_right = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][1], polygon_wkt.exterior.xy[1][1])
            bottom_left = to_planar_transformer.transform(polygon_wkt.exterior.xy[0][0], polygon_wkt.exterior.xy[1][0])

            # Compute limits
            x_min = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
            x_max = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
            y_min = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
            y_max = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])

            print(f'- Clipping roads for grid cell {idx}')

            # Compose shapely polygon for the current grid tile
            grid_polygon = Polygon([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])

            # Clip roads by grid polygon
            roads_clipped = gpd.clip(roads_buffered, grid_polygon)

            print(f"  Done, got {len(roads_clipped)} features")

            if roads_clipped.empty:
                roads_clipped_wkt = MultiPolygon([roads_clipped.any()]).wkt
            elif roads_clipped.geom_type[0] == 'Polygon':
                roads_clipped_wkt = MultiPolygon([roads_clipped.all()]).wkt
            elif roads_clipped.geom_type[0] == 'MultiPolygon':
                roads_clipped_wkt = MultiPolygon(roads_clipped.all()).wkt
            else:
                raise ValueError('Unknown geometry type')

            # Find raster images for the give grid tile
            images = sorted(glob.glob(os.path.join(images_dir, f'{idx}', '**', '**', '*.tif')))

            for image_filename in images:

                image_name = os.path.splitext(os.path.basename(image_filename))[0]

                location = os.path.dirname(grid_csv).split(os.sep)[-1]
                timestamp = image_name.split('_')[0]
                satellite = image_name.split('_')[1]
                resolution = image_name.split('_')[3]

                band_name = f'{satellite}_{resolution}'
                out_image_name = f'{location}_{timestamp}_{idx}'

                if not os.path.exists(os.path.join(output_dir, band_name)):
                    os.makedirs(os.path.join(output_dir, band_name))

                out_filename = os.path.join(output_dir, band_name, f'{out_image_name}.tif')

                # Clip image around grid polygon
                if not os.path.exists(out_filename):
                    subprocess.check_call(['gdalwarp', '-te', str(x_min), str(y_min), str(x_max), str(y_max),
                                           image_filename, out_filename], stderr=subprocess.STDOUT)

                if out_image_name not in grid_sizes_rows:
                    grid_sizes_rows[out_image_name] = (out_image_name, x_min, x_max, y_min, y_max)

                if out_image_name not in train_wkt_rows:
                    train_wkt_rows[out_image_name] = (out_image_name, 1, roads_clipped_wkt)

    grid_sizes_csv = os.path.join(output_dir, 'grid_sizes.csv')
    print(f"- Saving grid sizes in CSV format to: {grid_sizes_csv}")
    with open(grid_sizes_csv, "wt", encoding="utf-8", newline="") as grid_sizes_csv_file:
        writer = csv.DictWriter(grid_sizes_csv_file, fieldnames=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'])
        writer.writeheader()
        for image_id, x_min, x_max, y_min, y_max in list(grid_sizes_rows.values()):
            writer.writerow({'ImageId': image_id, 'Xmax': x_max, 'Ymin': y_min, 'Xmin': x_min, 'Ymax': y_max})
    print("  Done")

    train_wkt_csv = os.path.join(output_dir, 'train_wkt_v4.csv')
    print(f"- Saving WKT train polygons in CSV format to: {train_wkt_csv}")
    with open(train_wkt_csv, "wt", encoding="utf-8", newline="") as train_wkt_csv_file:
        writer = csv.DictWriter(train_wkt_csv_file, fieldnames=['ImageId', 'ClassType', 'MultipolygonWKT'])
        writer.writeheader()
        for image_id, class_type, multipolygon_wkt in list(train_wkt_rows.values()):
            writer.writerow({'ImageId': image_id, 'ClassType': class_type, 'MultipolygonWKT': multipolygon_wkt})
    print("  Done")


def main():
    grid_file = PARAMS.grid_file
    roads_file = PARAMS.roads_file
    crs = PARAMS.crs

    _compose_dataset(grid_file, roads_file, crs)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to download patches from a grid of polygons in WKT format")
    parser.add_argument(
        "--grid_file",
        help="Grid file with polygons of the areas downloaded",
        required=True
    )
    parser.add_argument(
        "--roads_file",
        help="Roads shape file",
        required=True
    )
    parser.add_argument(
        "--crs",
        help="Coordinate Reference System to use",
        default="epsg:3978"
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
