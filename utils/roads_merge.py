import argparse
import csv
import geopandas as gpd
import os

from grid_merge import _compute_limits
from shapely.geometry import MultiPolygon, Polygon


def _create_wkt(roads_shp, x_min, x_max, y_min, y_max, crs):

    data_dir = os.path.dirname(roads_shp)

    print("- Loading roads shapefile")

    roads = gpd.read_file(roads_shp)
    roads = roads.to_crs(crs)

    # No need to filter roads since the shape file is expected to have all the right features
    # roads_filtered = roads[[a or b or c for a, b, c in zip(roads.highway == 'primary',
    #                                                        roads.highway == 'secondary',
    #                                                        roads.highway == 'trunk')]]

    roads_selected = roads[['geometry']]

    print(f"  Done, got {len(roads)} features of type {roads_selected.geom_type[0]}")

    print("- Dissolving roads")

    roads_dissolved = roads_selected.dissolve()

    print(f"  Done, got {len(roads_dissolved)} features of type {roads_dissolved.geom_type[0]}")

    print("- Creating buffer around roads")

    roads_buffered = roads_dissolved.buffer(10)

    print(f"  Done, got {len(roads_buffered)} features of type {roads_buffered.geom_type[0]}")

    print("- Clipping roads buffered lines")

    grid_polygon = Polygon([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])

    roads_clipped = gpd.clip(roads_buffered, grid_polygon)

    # roads_clipped_gdf = gpd.GeoDataFrame(geometry=roads_clipped)
    # roads_clipped_gdf.to_file("{}/{}".format(data_dir, 'highway_thompson_clipped.shp'))

    roads_wkt_csv = '{}/{}'.format(data_dir, 'roads_wkt.csv')

    print(f"  Done, got {len(roads_clipped)} features of type {roads_clipped.geom_type[0]}")

    print(f"- Saving WKT clipped roads polygons in CSV format to: {roads_wkt_csv}")

    roads_clipped_wkt = MultiPolygon([roads_clipped.all()]).wkt

    with open(roads_wkt_csv, "wt", encoding="utf-8", newline="") as roads_wkt_csv_file:
        writer = csv.DictWriter(roads_wkt_csv_file, fieldnames=["ImageId", "ClassType", "MultipolygonWKT"])
        writer.writeheader()
        writer.writerow({"ImageId": "city", "ClassType": "1", "MultipolygonWKT": roads_clipped_wkt})

    print("  Done")


def main():
    roads_file = PARAMS.roads_file
    grid_file = PARAMS.grid_file
    crs = PARAMS.crs

    # TODO Fix limits computation, due to coordinates projection clipping is not so correct
    x_min, x_max, y_min, y_max = _compute_limits(grid_file, crs)

    _create_wkt(roads_file, x_min, x_max, y_min, y_max, crs)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to merge roads features")
    parser.add_argument(
        "--roads_file",
        help="Roads Shape file with polygons of roads to merge",
        required=True
    )
    parser.add_argument(
        "--grid_file",
        help="Grid CSV file with polygons of the areas to download",
        required=True
    )
    parser.add_argument(
        "--crs",
        help="Output coordinate reference system",
        default="epsg:3978"
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
