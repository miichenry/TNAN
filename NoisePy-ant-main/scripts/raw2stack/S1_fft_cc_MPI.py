"""
This main script of NoisePy:
    1) read the saved noise data in user-defined chunk of inc_hours, cut them into smaller length segments, do
    general pre-processing (trend, normalization) and then do FFT;
    2) save all FFT data of the same time chunk in memory;
    3) performs cross-correlation for all station pairs in the same time chunk and output the sub-stack (if
    selected) into ASDF format;
"""
import gc
import sys
import time
import pyasdf
import os, glob
import numpy as np
from mpi4py import MPI
from scipy.fftpack.helper import next_fast_len
import yaml
sys.path.insert(0, '/srv/beegfs/scratch/shares/cdff/DPM/NANT/NoisePy-ant-main/noisepy')
import cross_correlation

# ignore warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


tt0 = time.time()

# PARAMETER SECTION
#config_file = 'S1_params.yaml'#sys.argv[1]  # Input parameter file as first argument
config_file = sys.argv[1]  # Input parameter file as first argument
with open(config_file, 'r') as file:
    fc_para = yaml.safe_load(file)

# absolute path parameters
rootpath = fc_para['rootpath']  # root path for this data processing
CCFDIR = fc_para['CCFDIR']  # dir to store CC data
DATADIR = fc_para['DATADIR']  # dir where noise data is located
local_data_path = fc_para['local_data_path']  # absolute dir where SAC files are stored: this para is VERY IMPORTANT and has to be RIGHT if input_fmt is not h5 for asdf!!!
locations = fc_para['locations']  # station info including

# some control parameters
input_fmt = fc_para['input_fmt']  # string: 'h5', 'sac','mseed'
freq_norm = fc_para['freq_norm']  # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
time_norm = fc_para['time_norm']  # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
cc_method = fc_para['cc_method']  # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
flag = fc_para['flag']  # print intermediate variables and computing time for debugging purpose
acorr_only = fc_para['acorr_only'] # only perform auto-correlation
ncomp = fc_para['ncomp']  # 1 or 3 component data (needed to decide whether do rotation)

# station/instrument info for input_fmt=='sac' or 'mseed'
stationxml = fc_para['stationxml']  # station.XML file used to remove instrument response for SAC/miniseed data
rm_resp = fc_para['rm_resp']  # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
respdir = fc_para['respdir']  # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')

# read station list
if input_fmt != 'h5':
    if not os.path.isfile(locations):
        raise ValueError('Abort! station info is needed for this script')
    locs = pd.read_csv(locations,converters={'station':str})

# pre-processing parameters
cc_len = fc_para['cc_len']  # basic unit of data length for fft (sec)
step = fc_para['step']  # time step (sec). Overlap between each cc_len is cc_len-step
smooth_N = fc_para['smooth_N']  # moving window length for time/freq domain normalization if selected (points)

# cross-correlation parameters
maxlag = fc_para['maxlag']  # lags of cross-correlation to save (sec)
substack = fc_para['substack']  # True = smaller stacks within the time chunk. False: it will stack over inc_hours
# for instance: substack=True, substack_len=cc_len means that you keep ALL of the correlations
# if substack=True, substack_len=2*cc_len, then you pre-stack every 2 correlation windows.
substack_len = fc_para['substack_len']  # how long to stack over (for monitoring purpose): need to be multiples of cc_len
smoothspect_N = fc_para['smoothspect_N']   # moving window length to smooth spectrum amplitude (points)

# criteria for data selection
max_over_std = fc_para['max_over_std']  # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

# maximum memory allowed per core in GB
MAX_MEM = fc_para["MAX_MEM"]

# load useful download info if start from ASDF
dfile = os.path.join(DATADIR, 'download_info.yaml')
with open(dfile, "r") as file:
    down_info = yaml.safe_load(file)
samp_freq = down_info['samp_freq']
freqmin = down_info['freqmin']
freqmax = down_info['freqmax']
start_date = down_info['start_date']
end_date = down_info['end_date']
inc_hours = down_info['inc_hours']
ncomp = down_info['ncomp']

# Add down_info parameters to fc_para
fc_para.update(down_info)

dt = 1 / samp_freq
fc_para['dt'] = dt
##################################################
# save fft metadata for future reference
fc_metadata = os.path.join(CCFDIR, 'fft_cc_data.yaml')
if os.path.exists(fc_metadata):
    fc_metadata = fc_metadata.replace(".yaml", "_.yaml")

# --------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not os.path.isdir(CCFDIR): os.mkdir(CCFDIR)

    # save metadata
    with open(fc_metadata, 'w') as file:
        yaml.dump(fc_para, file, sort_keys=False)

    # set variables to broadcast
    tdir = sorted(glob.glob(os.path.join(DATADIR, '*.h5')))
    nchunk = len(tdir)
    splits = nchunk
    if nchunk == 0:
        raise IOError('Abort! no available seismic files for FFT')
else:
    splits, tdir = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits, root=0)
tdir = comm.bcast(tdir, root=0)

# MPI loop: loop through each user-defined time chunk
for ick in range(rank, splits, size):
    t10 = time.time()

    #############LOADING NOISE DATA AND DO FFT##################

    # get the tempory file recording cc process
    tmpfile = os.path.join(CCFDIR, tdir[ick].split('/')[-1].split('.')[0] + '.tmp')

    # check whether time chunk been processed or not
    if os.path.isfile(tmpfile):
        ftemp = open(tmpfile, 'r')
        alines = ftemp.readlines()
        if len(alines) and alines[-1] == 'done':
            Logger.info("Time chunk already processed. Next!")
            continue
        else:
            Logger.info("Time chunk already processed but not completed. Re-processing!")
            ftemp.close()
            os.remove(tmpfile)

    # retrieve station information
    ds = pyasdf.ASDFDataSet(tdir[ick], mpi=False, mode='r')
    sta_list = ds.waveforms.list()
    nsta = ncomp * len(sta_list)
    Logger.info(f" Rank {rank} found {nsta} stations in total")

    if len(sta_list) == 0:
        Logger.info('continue! no data in %s' % tdir[ick])
        continue

    # crude estimation on memory needs (assume float32)
    nsec_chunk = inc_hours / 24 * 86400
    nseg_chunk = int(np.floor((nsec_chunk - cc_len) / step))
    npts_chunk = int(nseg_chunk * cc_len * samp_freq)
    memory_size = nsta * npts_chunk * 4 / 1024 ** 3
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (
            memory_size, MAX_MEM))

    nnfft = int(next_fast_len(int(cc_len * samp_freq)))
    # open array to store fft data/info in memory
    fft_array = np.zeros((nsta, nseg_chunk * (nnfft // 2)), dtype=np.complex64)
    fft_std = np.zeros((nsta, nseg_chunk), dtype=np.float32)
    fft_flag = np.zeros(nsta, dtype=np.int16)
    fft_time = np.zeros((nsta, nseg_chunk), dtype=np.float64)
    # station information (for every channel)
    station = []
    network = []
    channel = []
    clon = []
    clat = []
    location = []
    elevation = []

    # loop through all stations
    iii = 0
    for ista in range(len(sta_list)):
        tmps = sta_list[ista]

        # get station and inventory
        try:
            inv1 = ds.waveforms[tmps]['StationXML']
        except Exception as e:
            Logger.warning('abort! no stationxml for %s in file %s' % (tmps, tdir[ick]))
            continue
        sta, net, lon, lat, elv, loc = cross_correlation.sta_info_from_inv(inv1)

        # get days information: works better than just list the tags
        all_tags = ds.waveforms[tmps].get_waveform_tags()
        if len(all_tags) == 0: continue

        # ----loop through each stream----
        for itag in range(len(all_tags)):
            if flag: Logger.info(f"Rank {rank} working on station {sta} and trace {all_tags[itag]}")

            # read waveform data
            source = ds.waveforms[tmps][all_tags[itag]]

            # channel info 
            comp = source[0].stats.channel
            print(f"test comp: {comp}")
            if comp[-1] == 'U': comp.replace('U', 'Z')
            if len(source) == 0: continue

            # cut daily-long data into smaller segments (dataS always in 2D)
            trace_stdS, dataS_t, dataS = cross_correlation.cut_trace_make_stat(fc_para,
                                                                               source)  # optimized version:3-4 times faster
            if not len(dataS): continue
            N = dataS.shape[0]

            # do normalization if needed
            source_white = cross_correlation.noise_processing(fc_para, dataS)
            Nfft = source_white.shape[1]
            Nfft2 = Nfft // 2
            # if flag:Logger.info('N and Nfft are %d (proposed %d),%d (proposed %d)' %(N,nseg_chunk,Nfft,nnfft))

            # keep track of station info to write into parameter section of ASDF files
            station.append(sta)
            network.append(net)
            channel.append(comp)
            clon.append(lon)
            clat.append(lat)
            location.append(loc)
            elevation.append(elv)

            # load fft data in memory for cross-correlations
            data = source_white[:, :Nfft2]
            fft_array[iii] = data.reshape(data.size)
            fft_std[iii] = trace_stdS
            fft_flag[iii] = 1
            fft_time[iii] = dataS_t
            iii += 1
            del trace_stdS, dataS_t, dataS, source_white, data

    del ds

    # check whether array size is enough
    if iii != nsta:
        Logger.warning('it seems some stations miss data in download step, but it is OKAY!')

    #############PERFORM CROSS-CORRELATION##################
    ftmp = open(tmpfile, 'w')
    # make cross-correlations 
    for iiS in range(iii):
        fft1 = fft_array[iiS]
        source_std = fft_std[iiS]
        sou_ind = np.where((source_std < fc_para['max_over_std']) & (source_std > 0) & (np.isnan(source_std) == 0))[0]
        if not fft_flag[iiS] or not len(sou_ind): continue

        t0 = time.time()
        # -----------get the smoothed source spectrum for decon later----------
        sfft1 = cross_correlation.smooth_source_spect(fc_para, fft1)
        sfft1 = sfft1.reshape(N, Nfft2)
        t1 = time.time()
        if flag: Logger.info('smoothing source takes %6.4fs' % (t1 - t0))

        # get index right for auto/cross correlation
        istart = iiS;
        iend = iii

        # -----------now loop III for each receiver B----------
        for iiR in range(istart, iend):
            if acorr_only:
                if (station[iiR] != station[iiS]): continue
            if flag:
                Logger.info('receiver: %s %s %s' % (station[iiR], network[iiR], channel[iiR]))
            if not fft_flag[iiR]: continue

            fft2 = fft_array[iiR]
            sfft2 = fft2.reshape(N, Nfft2)
            receiver_std = fft_std[iiR]

            # ---------- check the existence of earthquakes ----------
            rec_ind = \
                np.where((receiver_std < fc_para['max_over_std']) & (receiver_std > 0) & (np.isnan(receiver_std) == 0))[
                    0]
            bb = np.intersect1d(sou_ind, rec_ind)
            if len(bb) == 0:
                continue

            t2 = time.time()
            corr, tcorr, ncorr = cross_correlation.correlate(sfft1[bb, :], sfft2[bb, :], fc_para, Nfft,
                                                             fft_time[iiR][bb])
            t3 = time.time()

            # ---------------keep daily cross-correlation into a hdf5 file--------------
            tname = tdir[ick].split('/')[-1]
            cc_h5 = os.path.join(CCFDIR, tname)
            crap = np.zeros(corr.shape, dtype=corr.dtype)

            with pyasdf.ASDFDataSet(cc_h5, mpi=False) as ccf_ds:
                coor = {'lonS': clon[iiS],
                        'latS': clat[iiS],
                        'lonR': clon[iiR],
                        'latR': clat[iiR]}
                comp = channel[iiS][-1] + channel[iiR][-1]
                parameters = cross_correlation.cc_parameters(fc_para, coor, tcorr, ncorr, comp)

                # source-receiver pair
                data_type = network[iiS] + '.' + station[iiS] + '_' + network[iiR] + '.' + station[iiR]
                path = channel[iiS] + '_' + channel[iiR]
                crap[:] = corr[:]
                ccf_ds.add_auxiliary_data(data=crap,
                                          data_type=data_type,
                                          path=path,
                                          parameters=parameters)
                ftmp.write(network[iiS] + '.' + station[iiS] + '.' + channel[iiS] + '_' + network[iiR] + '.' + station[
                    iiR] + '.' + channel[iiR] + '\n')

            t4 = time.time()
            if flag:
                Logger.info('read S %6.4fs, cc %6.4fs, write cc %6.4fs' % ((t1 - t0), (t3 - t2), (t4 - t3)))

            del fft2, sfft2, receiver_std
        del fft1, sfft1, source_std

    # create a stamp to show time chunk being done
    ftmp.write('done')
    ftmp.close()

    fft_array = []
    fft_std = []
    fft_flag = []
    fft_time = []
    n = gc.collect()
    Logger.info(f"unreadable garbarge: {n}")

    t11 = time.time()
    Logger.info('it takes %6.2fs to process the chunk of %s' % (t11 - t10, tdir[ick].split('/')[-1]))

tt1 = time.time()
Logger.info('it takes %6.2fs to process step 1 in total' % (tt1 - tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
