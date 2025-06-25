#%%
from pynwb import NWBHDF5IO
file = '/data2/Projects/NKI_RS2/MoBI/NWB_BIDS/sub-M10901084/ses-MOBI1A/sub-M10901084_ses-MOBI1A_task-checkerboard_run-01_MoBI.nwb'
with NWBHDF5IO(file, 'r') as io:
    nwbfile = io.read()
    eeg = nwbfile.acquisition["ElectricalSeries"].data[:]
    electrodes = nwbfile.acquisition["ElectricalSeries"].electrodes[:]
    description = nwbfile.acquisition["ElectricalSeries"].description[:]
    stim = nwbfile.acquisition["StimLabels"].data[:]
    timestamps = nwbfile.acquisition["StimLabels"].timestamps[:]

# %%
