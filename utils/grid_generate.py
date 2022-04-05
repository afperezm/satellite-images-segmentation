import argparse
import csv
import os
import re
import string

import geopandas as gpd

from sentinelhub import BBox, UtmZoneSplitter
from shapely.geometry import Point, Polygon

FIELDNAMES = ['Id', 'PolygonWkt']
PARAMS = None


def _create_and_save_grid(aoi_geojson, grid_step_size, x_scale, y_scale, output_dir, crs):

    # TODO Use location name to retrieve geometry from OSM instead of processing a GeoJSON
    # location = location.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # location = re.sub(' +', ' ', location)
    # location = location.replace(' ', '_')
    # location = str.lower(location)

    # Use filename location name
    location = os.path.splitext(os.path.basename(aoi_geojson))[0]

    # Create output directory
    output_dir = os.path.join(output_dir, location)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load area of interest GeoJSON adn split into smaller bounding boxes
    aoi = gpd.read_file(aoi_geojson)
    aoi = aoi.to_crs(crs)
    print(x_scale, y_scale)
    aoi = aoi.scale(x_scale, y_scale)
    aoi = aoi.buffer(500)

    print("- Splitting area of interest into small bounding boxes of {} km2".format(grid_step_size / 1000))

    bbox_splitter = UtmZoneSplitter(shape_list=[aoi.geometry[0]],
                                    crs=aoi.crs,
                                    bbox_size=grid_step_size)

    bbox_list = bbox_splitter.get_bbox_list()

    print(f"  Done, got {len(bbox_list)} bounding boxes")

    # Convert between coordinate systems
    # https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
    # in_proj = pyproj.Proj(init='epsg:4326')
    # out_proj = pyproj.Proj(init='epsg:3978')

    # Website to check coordinate systems conversion
    # https://mygeodata.cloud/cs2cs/

    # Set up transformers
    # Planar coordinates: EPSG:3978 NAD83 / Canada Atlas Lambert
    # Geographical coordinates: EPSG:4326 WGS 84
    # to_planar_transformer = Transformer.from_crs('epsg:4326', crs, always_xy=True)
    # to_geographic_transformer = Transformer.from_crs(crs, 'epsg:4326', always_xy=True)

    # Transform list of bounding boxes into a list of polygons
    grid_polygons = []

    for bbox in bbox_list:

        top_left = (bbox.min_x, bbox.max_y)
        top_right = (bbox.max_x, bbox.max_y)
        bottom_right = (bbox.max_x, bbox.min_y)
        bottom_left = (bbox.min_x, bbox.min_y)

        polygon = Polygon([top_left, top_right, bottom_right, bottom_left])

        grid_polygons.append(polygon.wkt)

    grid_csv = os.path.join(output_dir, "grid_wkt.csv")

    print(f"- Saving WKT grid polygons in CSV format to: {grid_csv}")

    with open(grid_csv, "wt", encoding="utf-8", newline="") as grid_csv_file:
        grid_writer = csv.DictWriter(grid_csv_file, fieldnames=FIELDNAMES)
        grid_writer.writeheader()
        for idx, polygon_wkt in enumerate(grid_polygons):
            grid_writer.writerow({
                FIELDNAMES[0]: idx,
                FIELDNAMES[1]: polygon_wkt
            })

    print("  Done")


def main():
    output_dir = PARAMS.output_dir
    aoi_file = PARAMS.aoi_file
    grid_step_size = PARAMS.step_size  # Defaults to 5 km
    x_factor = PARAMS.x_factor
    y_factor = PARAMS.y_factor
    crs = PARAMS.crs  # Defaults to EPSG 3978

    _create_and_save_grid(aoi_file, grid_step_size, x_factor, y_factor, output_dir, crs)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to generate a grid around a location")
    parser.add_argument(
        "--output_dir",
        help='Output directory to store generated grid',
        required=True
    )
    parser.add_argument(
        "--aoi_file",
        help="Area of interest file (in GeoJSON format)",
        required=True
    )
    parser.add_argument(
        "--step_size",
        type=int,
        help="Bounding boxes size (in meters)",
        default=5000
    )
    parser.add_argument(
        "--x_factor",
        type=float,
        help="Horizontal scale factor",
        default=1.0
    )
    parser.add_argument(
        "--y_factor",
        type=float,
        help="Vertical scale factor",
        default=1.0
    )
    parser.add_argument(
        "--crs",
        help="Planar Coordinate Reference System used for grid generation",
        default="epsg:3978"
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
