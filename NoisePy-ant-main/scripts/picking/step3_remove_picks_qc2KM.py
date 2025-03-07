
"""
Discard picks outside two standard deviations
from the mean value of Vg at each period.

Genevieve Savard @ UniGe (last updated 04.08.2023)

Original codes by GS
:Location: Chavannes-pres-renens, CH
:Date: Aug 2023
:Author: K. Michailos
"""

import pandas as pd
import numpy as np
import os
import pandas
import numpy as np
from scipy.io import savemat
import logging
import pickle
import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


def picks_recursive_filtering(picks, multiplier=2):
    """
    Remove recursively picks outside a multiplier of the standard deviation around the mean at each period.
    Loop stops when all picks are within the defined boundaries.
    """
    df = picks.copy()
    Nremoved = 1000  # dummy initialization
    while Nremoved > 10:

        # Filter within 2 standard deviations
        groups_byt_gv = df.groupby('inst_period')['group_velocity']
        gv_mean = groups_byt_gv.transform('mean')
        gv_std = groups_byt_gv.transform('std')
        ikeep = df['group_velocity'].between(gv_mean.sub(gv_std.mul(multiplier)),
                                             gv_mean.add(gv_std.mul(multiplier)), inclusive=False)
        Nremoved = df.shape[0] - ikeep.sum()
        df = df.loc[ikeep, :]

    return df


if __name__ == "__main__":

    import platform  # KM
    import matplotlib

#    # Set up parameters and paths
#    if platform.node().startswith('kmichailos-laptop'):
#        hard_drive_dir = '/media/kmichailos/SEISMIC_DATA'
#        desktop_dir = '/home/kmichailos/Desktop'
#    else:
#        hard_drive_dir = '/media/konstantinos/SEISMIC_DATA'
#        desktop_dir = '/home/konstantinos/Desktop'
#
    #rootpath  = desktop_dir + "/ant_test"
    merged_pick_dir = "/srv/beegfs/scratch/shares/cdff/DPM/NANT/NoisePy-ant-main/scripts/picking/csv/picks_merged_DPM_vel_0.2_4.0_SNR1.0_lambda1.0.csv"
    output_dir = "/srv/beegfs/scratch/shares/cdff/DPM/NANT/NoisePy-ant-main/scripts/picking/csv/"
    # Read file with picks
    picks = pd.read_csv(merged_pick_dir)

    lag = "sym"
    pick_method = "topology"
    stack_method = "pws"
    score_thresh = 0.5
    number_of_std = 2.0
    ratio_d_lambda = 1.0
    comp = "ZZ"
    snr_nbG_thresh = 1.0

    df = picks.loc[(picks.pick_method==pick_method) &
                   (picks.score >= score_thresh) &
                   (picks.component==comp) &
                   (picks.stack_method==stack_method) &
                   (picks.lag==lag) &
                   (picks.snr_nbG >= snr_nbG_thresh) &
                   (picks.ratio_d_lambda >= ratio_d_lambda), :]

    # Filter within X standard deviations
    df_filtered = picks_recursive_filtering(df, multiplier=number_of_std)

    output_file = os.path.join(output_dir + f"filtered_picks_vel_0.2_4.0_{comp}_lamb{ratio_d_lambda}_std{number_of_std}_{stack_method}_{pick_method}"+".csv")

    df_filtered.to_csv(output_file)
    print(f"Output file written: {output_file}")
