"""
Extract metadata from the raw log files created by SmartSolo. Makes a plot showing:
- Temperature
- Latitude & Longitude
- Altitude
- eCompass north (careful! absolute value is wrong if node not calibrated...)
- Tilted angle
- Roll and Pitch angles

Input arguments:
> python extract_QC_stats.py path_to_your_DCCDATA_dir your_output_path "2024/03/01 00:00:00" "2024/04/20 00:00:00"
where 
DCCDATA_DIR is parent folder containing the subfolders for each serial number (453...)
OUTPUT_DIR is where to save figure (JPG format) and data tables (CSV)



Author: genevieve.savard@unige.ch
"""

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOCAL_TIME_ZONE = 'CET'


def get_temperature(fname):
    temps = []
    times = []
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("[Temperature"):
                dum = next(f).split()[-1].replace("\"","")
                t = pd.to_datetime(dum, utc=True, format="%Y/%m/%d,%H:%M:%S")
                temp = np.float32(next(f).split()[-1])
                times.append(t)
                temps.append(temp)

    df = pd.DataFrame({"time_UTC": times})
    df['time_local'] = df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)
    df['temperature'] = temps
    return df
    
    
def get_gps_info(fname):    
    times = []
    compass = []
    tilt = []
    roll = []
    pitch = []
    longitude = []
    latitude = []
    altitude = []
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("[GPS"):                
                status = next(f).split("=")[-1].strip() # GPS Status
                if status == "GPS Synchronization":
                    # print(line)
                    dum = next(f).split()[-1].strip("\"")
                    times.append(pd.to_datetime(dum, utc=True, format="%Y/%m/%d,%H:%M:%S"))
                    dum = next(f) # Lead Second
                    for k in range(15):                        
                        if dum.startswith("eCompass"):
                            compass.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Tilted Angle"):
                            tilt.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Roll Angle"):
                            roll.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Pitch Angle"):
                            pitch.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Longitude"):
                            longitude.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Latitude"):
                            latitude.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Altitude"):                        
                            try:
                                altitude.append(np.float32(dum.split()[-1]))                        
                            except:
                                altitude.append(np.nan)
                        if dum == "\n":
                            for val in [compass, tilt, roll, pitch, longitude, latitude, altitude]:
                                if len(val) < len(times):
                                    val.append(np.nan)
                            break
                        else:
                            try:
                                dum = next(f)
                            except:
                                for val in [compass, tilt, roll, pitch, longitude, latitude, altitude]:
                                    if len(val) < len(times):
                                        val.append(np.nan)
                                pass

    df = pd.DataFrame({"time_UTC": times})
    df['time_local'] = df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)
    df['compass'] = compass
    df["tilt"] = tilt
    df["roll"] = roll
    df["pitch"] = pitch
    df["longitude"] = longitude
    df["latitude"] = latitude
    df["altitude"] = altitude    
    return df
    
    
if __name__ == "__main__":
    
    DCCDATA_DIR = sys.argv[1] #'/Volumes/Indonesia/DCCDATA/MuaraLaboh_Supreme/Supreme1_2'
    OUTPUT_DIR = sys.argv[2] #'/Volumes/TOSHI/NANT/NoisePy-ant-main/smartsolo/QC' #
    start_date_str = sys.argv[3]  # e.g. "2024/03/01,00:00:00"
    end_date_str = sys.argv[4]  # e.g. "2024/04/25,00:00:00"
    start_date = pd.to_datetime(start_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    end_date = pd.to_datetime(end_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    print(f"Keeping data between {start_date_str} and {end_date_str}")
    
    
    filelist = glob.glob(os.path.join(DCCDATA_DIR,"*", "*", "DigiSolo.LOG")) #
    print(f"Number of DigiSolo.LOG files found: {len(filelist)}")
    filelist.sort()
    t_totals = []
    serial_numbers = []
    deploy_infos = []
    
    for fname in filelist:
        print(fname)
        
        # Extract metadata
        gps = get_gps_info(fname)
        
        # Keep only dates after start date
        gps = gps.loc[(gps.time_UTC >= start_date) & (gps.time_UTC <= end_date), :]

        # Get serial number
        n = len(DCCDATA_DIR.rstrip("/").split("/"))
        serial_number = fname.split("/")[n]
        
        # Calculate max days per station
        t_0 = min(gps.time_UTC)
        t_end = max(gps.time_UTC)
        t_total = int((t_end-t_0).days)
        t_totals.append(int(t_total))
        serial_numbers.append(int(serial_number))
        
        # Save data to CSV table
        dict = {'serial_number': serial_numbers, 'total days': t_totals}
        df = pd.DataFrame(dict)
        fname_table1 = os.path.join(OUTPUT_DIR, f"total_deploy_time.csv")
        df.to_csv(fname_table1, index=False)
