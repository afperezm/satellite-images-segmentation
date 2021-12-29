import argparse
import csv
import geopandas as gpd
import pandas as pd
import os

from shapely.geometry import MultiPolygon


def _create_wkt(roads_shp, grid_csv, crs):

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

    grid = pd.read_csv(grid_csv)

    grid_polygons = gpd.GeoSeries.from_wkt(grid['PolygonWkt'], crs='epsg:4326')
    grid_polygons = grid_polygons.to_crs(crs)
    grid_polygon = grid_polygons.unary_union

    roads_clipped = gpd.clip(roads_buffered, grid_polygon)

    # roads_clipped_gdf = gpd.GeoDataFrame(geometry=roads_clipped)
    # roads_clipped_gdf.to_file("{}/{}".format(data_dir, 'highway_thompson_clipped.shp'))

    roads_wkt_csv = '{}/{}'.format(data_dir, 'roads_wkt.csv')

    print(f"  Done, got {len(roads_clipped)} features of type {roads_clipped.geom_type[0]}")

    print(f"- Saving WKT clipped roads polygons in CSV format to: {roads_wkt_csv}")

    if roads_clipped.empty:
        roads_clipped_wkt = MultiPolygon([roads_clipped.any()]).wkt
    elif roads_clipped.geom_type[0] == 'Polygon':
        roads_clipped_wkt = MultiPolygon([roads_clipped.all()]).wkt
    elif roads_clipped.geom_type[0] == 'MultiPolygon':
        roads_clipped_wkt = MultiPolygon(roads_clipped.all()).wkt
    else:
        raise ValueError('Unknown geometry type')

    with open(roads_wkt_csv, "wt", encoding="utf-8", newline="") as roads_wkt_csv_file:
        writer = csv.DictWriter(roads_wkt_csv_file, fieldnames=["ImageId", "ClassType", "MultipolygonWKT"])
        writer.writeheader()
        writer.writerow({"ImageId": "city", "ClassType": "1", "MultipolygonWKT": roads_clipped_wkt})

    print("  Done")


def main():
    roads_file = PARAMS.roads_file
    grid_file = PARAMS.grid_file
    crs = PARAMS.crs

    _create_wkt(roads_file, grid_file, crs)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to merge roads features")
    parser.add_argument(
        "--roads_file",
        help="Roads shape file with polygons of roads to merge",
        required=True
    )
    parser.add_argument(
        "--grid_file",
        help="Grid CSV file (in WKT format) with polygons (coordinates in given CRS) of the areas to download",
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
