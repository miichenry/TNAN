"""
Extract metadata from the raw log files created by SmartSolo. Makes a csv:
- Latitude & Longitude
- Elevation
- Serial number


Input arguments:
> python extract_QC_stats.py DCCDATA_DIR OUTPUT_DIR START
where 
DCCDATA_DIR is parent folder containing the subfolders for each serial number (453...)
OUTPUT_DIR is where to save figure (JPG format) and data tables (CSV)

Author: michail.henry@etu.unige.ch
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOCAL_TIME_ZONE = 'CET'


def get_stats(fname):
    longitude = []
    latitude = []
    altitude = []
    times = []
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("Longitude"):
                try:
                    long = float(line.split()[-1])
                    lat = float(next(f).split()[-1])
                    alt = float(next(f).split()[-1])
                    dum = next(f).split()[-1].replace("\"", "")
                    t = pd.to_datetime(dum, utc=True, format="%Y/%m/%d,%H:%M:%S")
                    longitude.append(long)
                    latitude.append(lat)
                    altitude.append(alt)
                    times.append(t)
                except ValueError as e: ##
                    print(f"Error parsing line in {fname}: {e}")
                    continue

    df = pd.DataFrame({"time_UTC": times})
    df["longitude"] = longitude
    df['time_local'] = df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)
    df['latitude'] = latitude
    df['altitude'] = altitude
    return df

def serial_num(fname):
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if i == 5:
                SN = f"{line[16:-1]}"
                break
    return SN

if __name__ == "__main__":

    DCCDATA_DIR = sys.argv[1]  # DCCDATA folder path
    OUTPUT_DIR = sys.argv[2]  # Output directory for csv and figure files
    start_date_str = sys.argv[3]  # e.g. "2024/03/01,00:00:00"
    end_date_str = sys.argv[4]  # e.g. "2024/04/25,00:00:00"
    csv_name = sys.argv[5] # prefix of csv filename
    start_date = pd.to_datetime(start_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    end_date = pd.to_datetime(end_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    print(f"Keeping data between {start_date_str} and {end_date_str}")

    # filelist = glob.glob(os.path.join(DCCDATA_DIR, "*", "*", "DigiSolo.LOG"))
    filelist = glob.glob(os.path.join(DCCDATA_DIR, "DigiSolo*.LOG"))
    print(f"Reading data from directory: {DCCDATA_DIR}: Number of DigiSolo.LOG files found: {len(filelist)}")
    filelist.sort()
    mean_stats = pd.DataFrame({'SN': [],
                               'Longitude': [],
                            'Latitude': [],
                            'Elevation': []})
    for fname in filelist:
        print(f"fname={fname}")

        # Extract metadata
        stats = get_stats(fname)

        # Keep only dates after start_date
        stats = stats.loc[(stats.time_UTC >= start_date) & (stats.time_UTC <= end_date), :]

        # Get serial number
        n = len(DCCDATA_DIR.rstrip("/").split("/"))
        serial_number = serial_num(fname)
        # print(f"SN={serial_number}")

        # Calculate mean values
        mean_longitude = stats['longitude'].mean()
        mean_latitude = stats['latitude'].mean()
        mean_altitude = stats['altitude'].mean()

        # Append to mean_stats dataframe
        mean_stats = pd.concat([mean_stats, pd.DataFrame({'SN': [serial_number],
                                                        'Longitude': [mean_longitude],
                                                        'Latitude': [mean_latitude],
                                                        'Elevation': [mean_altitude]})], ignore_index=True)
    # Save mean values to CSV table
    fname_table2 = os.path.join(OUTPUT_DIR, f"{csv_name}_station_locations_noisepy.csv")
    mean_stats.to_csv(fname_table2, index=False)
