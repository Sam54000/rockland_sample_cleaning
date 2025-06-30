#%%
import pandas as pd
import numpy as np
import mne_bids
from pathlib import Path
import bids_explorer.architecture as arch
import mne
import eeg_research.preprocessing.tools.utils as utils
import channels_handling as ch
from mne_bids.dig import _write_dig_bids


#%%
architecture = arch.BidsArchitecture(
    root=Path("/data2/Projects/NKI_RS2/MoBI/"),
    datatype="raw",
    extension=".vhdr",
)
for file_id, element in architecture:
    try:
        raw_bv = mne.io.read_raw_brainvision(element['filename'])
        raw_bv, montage = ch.set_channel_montage(raw_bv)
        BidsPath = mne_bids.BIDSPath(
            root=element['root'] / "derivatives",
            subject=element['subject'],
            session=element['session'],
            run=element['run'],
            datatype="eeg",
        )
        print(BidsPath.fpath.parent)
        if BidsPath.fpath.parent.is_dir():
            _write_dig_bids(BidsPath, raw_bv, overwrite=True)
    except Exception as e:
        print(f"  !!!!!  ERROR  !!!!!  {e}")
        continue
# %%
