#%%
import mne
import pandas
import pyxdf
filename = "/data2/Projects/NKI_RS2/MoBI/sub-M10901084/ses-MOBI1A/lsl/sub-M10901084_ses-MOBI1A_task-passivepresent_run-01_lsl.xdf.gz"
eeg, header = pyxdf.load_xdf(filename)
# %%
