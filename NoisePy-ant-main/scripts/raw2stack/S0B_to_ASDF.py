"""
this script helps clean the sac/mseed files stored on your local machine in order to be connected
with the NoisePy package. it is similar to the script of S0A in essence.

by Chengxin Jiang, Marine Denolle (Jul.30.2019)

NOTE:
    0. MOST occasions you just need to change parameters followed with detailed explanations to run the script.
    1. In this script, the station of the same name but of different channels are treated as different stations.
    2. The bandpass function from obspy will output data in float64 format in default.
    3. For flexibilty to handle data of messy structures, the code loops through all sub-directory in RAWDATA and collects the
    starttime and endtime info. this enables us to find all data pieces located in each targeted time window. However, this process
    significaly slows down the code, particuarly for data of a big station list. we recommend to prepare a csv file (L48) that contains
    all sac/mseed file names with full path and their associated starttime/endtime info if possible. based on tests, this improves the
    efficiency of the code by 2-3 orders of magnitude.
"""

import sys
import glob
import os, gc
import obspy
import time
import pyasdf
import numpy as np
import pandas as pd
from mpi4py import MPI
import yaml

sys.path.insert(0, '/srv/beegfs/scratch/shares/cdff/DPM/NANT/NoisePy-ant-main/noisepy')
import preprocess_h5

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
os.system('export HDF5_USE_FILE=FALSE')

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

tt0 = time.time()

# PARAMETER SECTION
config_file = 'S0B_params.yaml'#sys.argv[1]  # Input parameter file as first argument
with open(config_file, 'r') as file:
    prepro_para = yaml.safe_load(file)

# data/file paths
rootpath = prepro_para['rootpath']   # absolute path for your project
RAWDATA = prepro_para['RAWDATA']  # os.path.join(rootpath,'RAW_DATA')                           # dir where mseed/SAC files are located
DATADIR = prepro_para['DATADIR']  # dir where cleaned data in ASDF format are going to be outputted
locations = prepro_para['locations']  # station info including network,station,channel,latitude,longitude,elevation

# useful parameters for cleaning the data
input_fmt = prepro_para['input_fmt']  # input file format between 'sac' and 'mseed'
samp_freq = prepro_para['samp_freq']  # targeted sampling rate
stationxml = prepro_para['stationxml']  # station.XML file exists or not
rm_resp = prepro_para['rm_resp']  # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
respdir = prepro_para['respdir']  # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')
freqmin = prepro_para['freqmin']  # pre filtering frequency bandwidth
freqmax = prepro_para['freqmax']  # note this cannot exceed Nquist freq
flag = prepro_para['flag']  # print intermediate variables and computing time

# having this file saves a tons of time: see L95-126 for why
wiki_file = prepro_para['wiki_file']  # file containing the path+name for all sac/mseed files and its start-end time
allfiles_path = prepro_para['allfiles_path']  # make sure all sac/mseed files can be found through this format
messydata = prepro_para['messydata'] # set this to False when daily noise data directory is stored in sub-directory of Event_year_month_day
ncomp = prepro_para['ncomp']

# targeted time range # Dec 5 2020 to Jan 5 2021
start_date = prepro_para['start_date']  # start date of local data
end_date = prepro_para['end_date']  # end date of local data
inc_hours = prepro_para['inc_hours']  # sac/mseed file length for a continous recording

# get rough estimate of memory needs to ensure it now below up in S1
cc_len = prepro_para['cc_len']  # basic unit of data length for fft (s)
step = prepro_para['step']  # overlapping between each cc_len (s)
MAX_MEM = prepro_para['MAX_MEM']  # maximum memory allowed per core in GB

##################################################

metadata = os.path.join(DATADIR, 'download_info.yaml')
if os.path.exists(metadata):
    metadata = metadata.replace(".yaml", "_.yaml")

# ---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
Logger.info(f"Rank {rank}, size {size}")
# -----------------------

if rank == 0:
    # make directory
    if not os.path.isdir(DATADIR):
        os.mkdir(DATADIR)
    if not os.path.isdir(RAWDATA):
        Logger.info('no path of %s exists for', RAWDATA)
        os.mkdir(RAWDATA)

    # check station list
    if not os.path.isfile(locations):
        raise ValueError('Abort! station info is needed for this script')
    locs = pd.read_csv(locations,dtype={'station':str})
#    locs = pd.read_csv(locations)
    nsta = len(locs)

    # output parameter info
    with open(metadata, 'w') as file:
        yaml.dump(prepro_para, file, sort_keys=False)

    # assemble timestamp info
    all_stimes, allfiles = preprocess_h5.make_timestamps(prepro_para)

    # all time chunk for output: loop for MPI
    all_chunk = preprocess_h5.get_event_list(start_date, end_date, inc_hours)
    splits = len(all_chunk) - 1
    if splits < 1: raise ValueError(
        'Abort! no chunk found between %s-%s with inc %s' % (start_date, end_date, inc_hours))

    # rough estimation on memory needs needed in S1 (assume float32 dtype)
    nsec_chunk = inc_hours / 24 * 86400
    nseg_chunk = int(np.floor((nsec_chunk - cc_len) / step)) + 1
    npts_chunk = int(nseg_chunk * cc_len * samp_freq)
    memory_size = nsta * npts_chunk * 4 / 1024 ** 3
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (
        memory_size, MAX_MEM))
else:
    #     splits,all_chunk,all_stimes,allfiles = [None for _ in range(4)]
    splits, all_chunk, all_stimes, allfiles, locs = [None for _ in range(5)]

# broadcast the variables
splits = comm.bcast(splits, root=0)
all_chunk = comm.bcast(all_chunk, root=0)
all_stimes = comm.bcast(all_stimes, root=0)
allfiles = comm.bcast(allfiles, root=0)
locs = comm.bcast(locs, root=0)

# MPI: loop through each time-chunk
for ick in range(rank, splits, size):
    # Check if already processed
    ff = os.path.join(DATADIR, all_chunk[ick] + 'T' + all_chunk[ick + 1] + '.h5') # output file
    if os.path.isfile(ff):  # Skip if processed
        continue

    t0 = time.time()

    # time window defining the time-chunk
    s1 = obspy.UTCDateTime(all_chunk[ick])
    s2 = obspy.UTCDateTime(all_chunk[ick + 1])
    date_info = {'starttime': s1, 'endtime': s2}
    time1 = s1 - obspy.UTCDateTime(1970, 1, 1)
    time2 = s2 - obspy.UTCDateTime(1970, 1, 1)
    Logger.info(f"Starttime = {s1} (time1 = {time1}), Endtime = {s2} (time2 = {time2})")

    # find all data pieces having data of the time-chunk
    indx1 = np.where((time1 >= all_stimes[:, 0]) & (time1 < all_stimes[:, 1]))[0]
    indx2 = np.where((time2 > all_stimes[:, 0]) & (time2 <= all_stimes[:, 1]))[0]
    indx3 = np.where((time1 <= all_stimes[:, 0]) & (time2 >= all_stimes[:, 1]))[0]
    indx4 = np.concatenate((indx1, indx2, indx3))
    indx = np.unique(indx4)
    if not len(indx):
        Logger.warning('continue! no data found between %s-%s' % (s1, s2))
        continue

    Logger.info(f"Process {rank} of {size} has {len(indx)} files in window: {s1} to {s2} ({time1} to {time2})")

    # trim down the sac/mseed file list with time in time-chunk
    tfiles = []
    for ii in indx:
        tfiles.append(allfiles[ii])

    # loop through station
    nsta = len(locs)
    for ista in range(nsta):

        # the station info:
        station = str(locs.iloc[ista]['station'])
        Logger.info(f"{station}")
        network = locs.iloc[ista]['network']
        comp = locs.iloc[ista]['channel']
        if flag: Logger.info(f"Rank {rank} is working on station {station} channel {comp}")

        # norrow down file list by using sta/net info in the file name
        ttfiles = [ifile for ifile in tfiles if station in ifile]
        Logger.info(f"Files for station {station}: {ttfiles}")
        if not len(ttfiles):
            Logger.info(f"No files found for {station}")
            continue
        tttfiles = [ifile for ifile in ttfiles if comp in ifile]
        if not len(tttfiles):
            Logger.info(f"No files found for {station}.{comp}")
            continue

        source = obspy.Stream()
        for ifile in tttfiles:
            try:
                tr = obspy.read(ifile)
                for ttr in tr:
                    source.append(ttr)
            except Exception as inst:
                Logger.info(inst);
                continue

        # jump if no good data left
        if not len(source): Logger.warning("trace read from file"); continue

        # make inventory to save into ASDF file
        t1 = time.time()
        inv1 = preprocess_h5.stats2inv(source[0].stats, prepro_para, locs=locs)
        tr = preprocess_h5.preprocess_raw(source, inv1, prepro_para, date_info)

        # jump if no good data left
        if not len(tr): Logger.warning("No trace left after pre-processing"); continue
        if np.all(tr[0].data == 0): print("Data all zeroes after pre-processing. skip"); print(len(tr)); continue

        t2 = time.time()
        if flag: Logger.info(f"pre-processing takes {t2 - t1:.2f}s")

        # ready for output
        ff = os.path.join(DATADIR, all_chunk[ick] + 'T' + all_chunk[ick + 1] + '.h5')
        if not os.path.isfile(ff):
            with pyasdf.ASDFDataSet(ff, mpi=False, compression="gzip-3", mode='w') as ds:
                pass

        with pyasdf.ASDFDataSet(ff, mpi=False, compression="gzip-3", mode='a') as ds:
            # add the inventory for all components + all time of this station
            try:
                ds.add_stationxml(inv1)
            except Exception:
                pass

            tlocation = str('00')
            new_tags = '{0:s}_{1:s}'.format(comp.lower(), tlocation.lower())
            ds.add_waveforms(tr, tag=new_tags)

    t3 = time.time()
    Logger.info('it takes ' + str(t3 - t0) + ' s to process ' + str(inc_hours) + 'h length in step 0B')

tt1 = time.time()
Logger.info('step0B takes ' + str(tt1 - tt0) + ' s')

comm.barrier()
if rank == 0:
    sys.exit()
