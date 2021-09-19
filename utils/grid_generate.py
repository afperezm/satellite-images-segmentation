import argparse
import csv
import os
import re
import string

import geocoder

from pyproj import Transformer
from shapely.geometry import Point, Polygon

FIELDNAMES = ['Id', 'PolygonWkt']
PARAMS = None


def _create_and_save_grid(location, grid_width, grid_step_size, output_dir, use_centroid, crs):

    print(f"- Geocoding location: {location}")
    g = geocoder.google(location, key='AIzaSyDNRG-2VaztgMhymInIKZN3LF8vO3nIDQM')

    print("- Geocoded location")
    print(f"  {g.latlng}")

    print("- Geocoded location bounding box")
    print(f"  {g.bbox}")

    location = location.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    location = re.sub(' +', ' ', location)
    location = location.replace(' ', '_')
    location = str.lower(location)

    output_dir = os.path.join(output_dir, location)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert between coordinate systems
    # https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
    # in_proj = pyproj.Proj(init='epsg:4326')
    # out_proj = pyproj.Proj(init='epsg:3978')

    # Website to check coordinate systems conversion
    # https://mygeodata.cloud/cs2cs/

    # Set up transformers
    # Planar coordinates: EPSG:3978 NAD83 / Canada Atlas Lambert
    # Geographical coordinates: EPSG:4326 WGS 84
    to_planar_transformer = Transformer.from_crs('epsg:4326', crs, always_xy=True)
    to_geographic_transformer = Transformer.from_crs(crs, 'epsg:4326', always_xy=True)

    if use_centroid:
        centroid = Point(g.latlng[1], g.latlng[0])
        centroid_planar = to_planar_transformer.transform(centroid.x, centroid.y)

        northeast = to_geographic_transformer.transform(centroid_planar[0] + grid_width,
                                                        centroid_planar[1] + grid_width)
        southwest = to_geographic_transformer.transform(centroid_planar[0] - grid_width,
                                                        centroid_planar[1] - grid_width)

        bbox = {'northeast': [northeast[1], northeast[0]], 'southwest': [southwest[1], southwest[0]]}

        print("- Computed location")
        print(f"  {centroid}")

        print("- Computed location bounding box")
        print(f"  {bbox}")
    else:
        bbox = g.bbox

    top_right_corner = Point(bbox['northeast'][1], bbox['northeast'][0])
    bottom_left_corner = Point(bbox['southwest'][1], bbox['southwest'][0])

    # Project bounding box corners to planar coordinate system
    top_right_planar = to_planar_transformer.transform(top_right_corner.x, top_right_corner.y)
    bottom_left_planar = to_planar_transformer.transform(bottom_left_corner.x, bottom_left_corner.y)

    # Initialize list of grid
    grid_coordinates = []

    print("- Creating grid with rectangle polygons of {} km2".format(grid_step_size / 1000))

    # Loop from left to right
    x = bottom_left_planar[0]
    while top_right_planar[0] - x > 0.0001:
        # Loop from top to bottom
        y = top_right_planar[1]
        while y - bottom_left_planar[1] > 0.0001:
            top_left = to_geographic_transformer.transform(x, y)
            top_right = to_geographic_transformer.transform(x + grid_step_size, y)
            bottom_right = to_geographic_transformer.transform(x + grid_step_size, y - grid_step_size)
            bottom_left = to_geographic_transformer.transform(x, y - grid_step_size)
            polygon = Polygon([top_left, top_right, bottom_right, bottom_left])
            # print(f"  {polygon.wkt}")
            grid_coordinates.append(polygon.wkt)
            y -= grid_step_size
        x += grid_step_size

    print(f"  Done, got {len(grid_coordinates)} polygons")

    grid_csv = os.path.join(output_dir, "grid_wkt.csv")

    print(f"- Saving WKT grid polygons in CSV format to: {grid_csv}")

    with open(grid_csv, "wt", encoding="utf-8", newline="") as grid_csv_file:
        grid_writer = csv.DictWriter(grid_csv_file, fieldnames=FIELDNAMES)
        grid_writer.writeheader()
        for idx, polygon_wkt in enumerate(grid_coordinates):
            grid_writer.writerow({
                FIELDNAMES[0]: idx,
                FIELDNAMES[1]: polygon_wkt
            })

    print("  Done")


def main():
    output_dir = PARAMS.output_dir
    location = PARAMS.location
    grid_width = PARAMS.width
    grid_step_size = PARAMS.step_size  # Defaults to 10 km
    use_centroid = PARAMS.use_centroid
    crs = PARAMS.crs

    _create_and_save_grid(location, grid_width, grid_step_size, output_dir, use_centroid, crs)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to generate a grid around a location")
    parser.add_argument(
        "--output_dir",
        help='Output directory to store generated grid',
        required=True
    )
    parser.add_argument(
        "--location",
        help="Location name",
        required=True
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Grid width (in meters)",
        default=3000
    )
    parser.add_argument(
        "--step_size",
        type=int,
        help="Grid step size (in meters)",
        default=1000
    )
    parser.add_argument(
        "--use_centroid",
        action="store_true",
        help="Flag to indicated whether bounding box must be computed from geocoded centroid"
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
