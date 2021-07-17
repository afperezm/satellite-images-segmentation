import argparse
import csv
import datetime
import os

from coastsat import SDS_download, SDS_tools
from shapely import wkt
from shapely.geometry import Polygon

FIELDNAMES = ['Id', 'PolygonWkt']
PARAMS = None


def _download_images(start_date, end_date, satellites_list, grid_csv):

    output_dir = os.path.dirname(grid_csv)

    polygon_dumps = []

    with open(grid_csv, "rt", encoding="utf-8", newline="") as grid_csv_file:

        # Init CSV reader
        reader = csv.DictReader(grid_csv_file, fieldnames=FIELDNAMES)

        # Skip first CSV row
        next(reader)

        # Loop over all CSV rows
        for idx, row in enumerate(reader):

            # Retrieve polygon string in WKT format
            polygon_str = row[FIELDNAMES[1]]

            # Load polygons in WKT format from string
            polygon_wkt = wkt.loads(polygon_str)

            # Build polygon list
            polygon = [[[polygon_wkt.exterior.xy[0][0], polygon_wkt.exterior.xy[1][0]],
                        [polygon_wkt.exterior.xy[0][1], polygon_wkt.exterior.xy[1][1]],
                        [polygon_wkt.exterior.xy[0][2], polygon_wkt.exterior.xy[1][2]],
                        [polygon_wkt.exterior.xy[0][3], polygon_wkt.exterior.xy[1][3]]]]

            # Compose parameters to download images
            polygon = SDS_tools.smallest_rectangle(polygon)
            dates = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            sat_list = satellites_list
            site_name = f'{idx}'

            # Append polygon envelope with sides parallel to coordinate axes
            polygon_dumps.append(wkt.dumps(Polygon(polygon[0])))

            # Compose dictionary of parameters
            inputs = {'polygon': polygon,
                      'dates': dates,
                      'sat_list': sat_list,
                      'sitename': site_name,
                      'filepath': output_dir}

            # SDS_download.check_images_available(inputs)
            SDS_download.retrieve_images(inputs)

    # Save list of polygon envelopes
    grid_wkt_smallest_rectangle = "{}/{}".format(output_dir, "grid_wkt_smallest_rectangle.csv")

    print(f"- Saving WKT grid adjusted polygons in CSV format to: {grid_wkt_smallest_rectangle}")

    if os.path.exists(grid_wkt_smallest_rectangle):
        print(f"  Skipping, already saved")
        return

    with open(grid_wkt_smallest_rectangle, "wt", encoding="utf-8", newline="") as grid_wkt_smallest_rectangle_file:
        writer = csv.DictWriter(grid_wkt_smallest_rectangle_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for polygon_id, polygon_str in enumerate(polygon_dumps):
            writer.writerow({
                "Id": polygon_id,
                "PolygonWkt": polygon_str
            })


def main():
    start_date = PARAMS.start_date
    end_date = PARAMS.end_date
    satellites_list = PARAMS.satellites_list
    grid_file = PARAMS.grid_file

    _download_images(start_date, end_date, satellites_list, grid_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Utility to download patches from a grid of polygons in WKT format")
    parser.add_argument(
        "--grid_file",
        help="Grid file with polygons of the areas to download",
        required=True
    )
    parser.add_argument(
        "--start_date",
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        help="Start date",
        required=True
    )
    parser.add_argument(
        "--end_date",
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        help="End date",
        required=True
    )
    parser.add_argument(
        "--satellites_list",
        nargs="+",
        default=["S2"],
        help="List of satellite missions"
    )
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
