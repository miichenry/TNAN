import os
import glob
import pandas as pd

stainfo = pd.read_csv('/srv/beegfs/scratch/shares/cdff/DPM/NANT/coordinates_smartsolo.csv')
datadir = '/srv/beegfs/scratch/shares/cdff/DPM/georeva'
network = "KD"

for SN, station in zip(stainfo.serial_number.astype(str), stainfo.station.astype(str)):
    print(station,SN)
    olddir = os.path.join(datadir)    
    # Change file names 
    # Assumes SmartSolo export file pattern like 453007143.11.2023.06.26.00.00.00.000.E.miniseed (starts with serial number)
    flist = glob.glob(os.path.join(olddir, f"{SN}.*"))
    for oldfile in flist:
        print(oldfile)
        if ".Z." in oldfile:
            comp = "DPZ"
            newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
            os.rename(oldfile,newfile)
        elif ".N." in oldfile:
            comp = "DPN"
            newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
            os.rename(oldfile,newfile)
        elif ".E." in oldfile:
            comp = "DPE"
            newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
            os.rename(oldfile,newfile)
