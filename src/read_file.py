#%%
from pathlib import Path
import numpy as np
import mne
import pyxdf
from datetime import datetime
from pynwb import NWBHDF5IO
import os


def convert_signal(eeg_stream:dict) -> np.ndarray:
    units = {
        "microvolts": 10e-6,
        "millivolts": 10e-3,
        "volts": 1,
    }
    unit_matrix = list()
    chan_dict = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    signals = eeg_stream['time_series'].T
    unit_matrix = np.array([[units.get(chan['unit'][0].lower(),1) 
                   for chan in chan_dict]]).T
    return np.multiply(signals,unit_matrix)

def read_raw_xdf(filename: str | Path):
    eeg, header = pyxdf.load_xdf(filename, select_streams=5)
    sfreq = float(eeg[0]['info']['nominal_srate'][0])
    info = mne.create_info(**parse_channel_names(eeg[0]),sfreq=sfreq)
    return mne.io.RawArray(convert_signal(eeg[0]), info = info)

def read_raw_nwb(filename: str | os.PathLike) -> mne.io.Raw:

    with NWBHDF5IO(filename, 'r') as io:
        nwbfile = io.read()
        eeg_data = nwbfile.acquisition["ElectricalSeries"].data[:]
        eeg_time = nwbfile.acquisition["ElectricalSeries"].timestamps[:]
        electrodes = nwbfile.acquisition["ElectricalSeries"].electrodes[:]
        elec_names = nwbfile.acquisition["ElectricalSeries"].description[:]
        events_name = nwbfile.acquisition["StimLabels"].data[:]
        events_onset = nwbfile.acquisition["StimLabels"].timestamps[:]
        
    datetime_times = [
        datetime.fromtimestamp(t) for t in eeg_time
    ]
    first_time_step = datetime_times[1] - datetime_times[0]
    sfreq = 1e6 / first_time_step.microseconds

    info = mne.create_info(
        ch_names = elec_names.split(","),
        sfreq = sfreq,
        ch_types = "eeg"
    )

    raw = mne.io.RawArray(eeg_data.T*1e-6, info)

    coord = {
    item["group_name"].split(" ")[-1]: (
        item["x"],item["y"],item["z"] 
        )
    for _, item in electrodes.iterrows()
    }
    
    

    raw.set_meas_date(eeg_time[0])
    montage = mne.channels.make_dig_montage(ch_pos = coord)
    raw.set_montage(montage)

    eeg_start = datetime.fromtimestamp(eeg_time[0])
    delta = [datetime.fromtimestamp(t) - eeg_start for t in events_onset]
    onset = [d.total_seconds() for d in delta]
    duration = np.zeros_like(onset)

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=[name[0] for name in events_name],
        orig_time=raw.info["meas_date"]
    )
    
    raw.set_annotations(annotations)

    return raw
# %%
