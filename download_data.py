"""
Downloads the UCI Gas Turbine CO and NOx Emission dataset (real hourly
sensor measurements, 2011-2015, CC BY 4.0, DOI 10.24432/C5WC95).

The files are fetched from a GitHub mirror of the UCI data (checked to
match the original UCI files: same row counts per year, same columns,
same values).
"""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
YEARS = [2011, 2012, 2013, 2014, 2015]
BASE_URL = "https://raw.githubusercontent.com/kotyasmol/gas-turbine-co-regression-lab/main/data/raw"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for year in YEARS:
        filename = f"gt_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        dest = DATA_DIR / filename
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  saved {dest} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
