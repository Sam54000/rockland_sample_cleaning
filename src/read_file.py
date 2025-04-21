from pathlib import Path
import numpy as np
import mne
import pyxdf


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