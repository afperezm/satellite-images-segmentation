# Permafrost Imagery Segmentation

## Requirements

QGIS is required for visualization of geographical data, since it uses gdal it provides gdal system requirements 
(gdal-bin and libgdal-dev).

CoastSat is required for data download. Execute following steps for installation:

- clone github repo:
  - `git clone git@github.com:kvos/CoastSat.git`
- create new venv:
  - `python3 -m venv ~/sits-venv/`
- install latest wheel and pip:
  - `pip install --upgrade wheel pip`
- install gdal system requirements:
  - `sudo add-apt-repository ppa:ubuntugis/ppa`
  - `sudo apt-get install gdal-bin`
  - find the GDAL version to install the correct python bindings:
    - `ogrinfo --version`
  - sudo apt-get install libgdal-dev
- install gdal:
  - `export CPLUS_INCLUDE_PATH=/usr/include/gdal`
  - `export C_INCLUDE_PATH=/usr/include/gdal`
  - `pip install GDAL==<GDAL VERSION FROM OGRINFO>`
- install other requirements:
  - `pip install earthengine-api google-api-python-client pandas geopandas numpy matplotlib pillow pytz scikit-image scikit-learn shapely scipy astropy spyder notebook`
- activate google earth engine python API:
  - `earthengine authenticate`

## Usage

Make coastsat visible as package:
  - export PYTHONPATH=../CoastSat/

Generate grids:

```bash
python utils/grid_generate.py --output_dir ~/data/permafrost-imagery/ --location "Gillam, MB, Canada" --width 5000 --step_size 2000 --use_centroid --crs epsg:32615
```

```bash
python utils/grid_generate.py --output_dir ~/data/permafrost-imagery/ --location "Thompson, MB, Canada" --width 5000 --step_size 2000 --use_centroid --crs epsg:32614
```

```bash
python utils/grid_generate.py --output_dir ~/data/permafrost-imagery/ --location "Yellowknife, NT, Canada" --width 5000 --step_size 2000 --use_centroid --crs epsg:32612
```

Download images:

```bash
python utils/grid_download.py --grid_file ~/data/permafrost-imagery/gillam_mb_canada/grid_wkt.csv --start_date 2021-01-01 --end_date 2021-01-31 --satellites_list S2
```

```bash
python utils/grid_download.py --grid_file ~/data/permafrost-imagery/thompson_mb_canada/grid_wkt.csv --start_date 2021-01-01 --end_date 2021-01-31 --satellites_list S2
```

```bash
python utils/grid_download.py --grid_file ~/data/permafrost-imagery/yellowknife_nt_canada/grid_wkt.csv --start_date 2021-01-01 --end_date 2021-01-31 --satellites_list S2
```

Merge images:

```bash
python utils/grid_merge.py --grid_file ~/data/permafrost-imagery/gillam_mb_canada/grid_wkt.csv --satellites_list S2 --resolution_list 10m
```

```bash
python utils/grid_merge.py --grid_file ~/data/permafrost-imagery/thompson_mb_canada/grid_wkt.csv --satellites_list S2 --resolution_list 10m
```

```bash
python utils/grid_merge.py --grid_file ~/data/permafrost-imagery/yellowknife_nt_canada/grid_wkt.csv --satellites_list S2 --resolution_list 10m
```

Merge roads:

```bash
python utils/roads_merge.py --roads_file /home/andresf/data/permafrost-imagery/gillam_mb_canada/roads.shp --grid_file /home/andresf/data/permafrost-imagery/gillam_mb_canada/grid_wkt.csv --crs epsg:32615
```

```bash
python utils/roads_merge.py --roads_file /home/andresf/data/permafrost-imagery/thompson_mb_canada/roads.shp --roads_file --grid_file /home/andresf/data/permafrost-imagery/thompson_mb_canada/grid_wkt.csv --crs epsg:32614
```

```bash
python utils/roads_merge.py --roads_file /home/andresf/data/permafrost-imagery/yellowknife_nt_canada/roads.shp --roads_file --grid_file /home/andresf/data/permafrost-imagery/yellowknife_nt_canada/grid_wkt.csv --crs epsg:32612
```

Generate datasets:

```bash
python utils/dataset_generate.py --grid_file /home/andresf/data/permafrost-imagery/gillam_mb_canada/grid_wkt.csv --roads_file /home/andresf/data/permafrost-imagery/gillam_mb_canada/roads.shp --crs epsg:32615
```

```bash
python utils/dataset_generate.py --grid_file /home/andresf/data/permafrost-imagery/thompson_mb_canada/grid_wkt.csv --roads_file /home/andresf/data/permafrost-imagery/thompson_mb_canada/roads.shp --crs epsg:32614
```

```bash
python utils/dataset_generate.py --grid_file /home/andresf/data/permafrost-imagery/yellowknife_nt_canada/grid_wkt.csv --roads_file /home/andresf/data/permafrost-imagery/yellowknife_nt_canada/roads.shp --crs epsg:32612
```

## References

[1] https://qgis.org/en/site/forusers/alldownloads.html#debian-ubuntu

[2] https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
