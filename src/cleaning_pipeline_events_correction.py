#%%
from pathlib import Path
import os
nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import numpy as np
import mne
import pyprep
from datetime import datetime, timezone
import pytz
import mne_bids
import channels_handling as ch
import pandas as pd
import bids_explorer.architecture as arch
import eeg_research.preprocessing.tools.utils as utils

import pyxdf

def convert_in_volts(eeg_stream: dict) -> np.ndarray:
    """Convert the EEG from the XDF object in volts.
    From the units information contains in the XDF object, the EEG signal
    is converted in volts.

    Args:
        eeg_stream (dict): The stream in the XDF object dedicated to the EEG.

    Returns:
        np.ndarray: The EEG signal converted.
    """
    units = {
        "microvolts": 1e-6,
        "millivolts": 1e-3,
        "volts": 1,
    }
    unit_matrix = list()
    chan_dict = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    signals = eeg_stream["time_series"].T
    for chan in chan_dict:
        if chan.get("unit"):
            unit_matrix.append(units.get(chan["unit"][0].lower(), 1))

    unit_matrix = np.array(unit_matrix)[:, np.newaxis]

    return np.multiply(signals, unit_matrix)

def extract_streams(filename):
    streams_dict = {
        "eeg": pyxdf.load_xdf(filename, select_streams=[{"type":"EEG"}])[0][0],
        "markers": pyxdf.load_xdf(filename, select_streams=[{"name":"StimLabels"}])[0][0]
    }
    return streams_dict

def read_raw_xdf(streams_dict: dict) -> mne.io.RawArray:
    """Read the XDF file and convert it into an mne.io.RawArray.

    Args:
        filename (str | Path): The input filename to read.

    Returns:
        mne.io.RawArray: The mne raw object.
    """
    eeg = streams_dict["eeg"]
    sfreq = float(eeg["info"]["nominal_srate"][0])
    info = mne.create_info(**ch.parse_lsl_channel_names(eeg), sfreq=sfreq)
    raw = mne.io.RawArray(convert_in_volts(eeg), info=info)
    timestamp = eeg["time_stamps"][0]
    dt_naive = datetime.fromtimestamp(timestamp)
    eastern_tz = pytz.timezone('US/Eastern')
    dt_eastern = eastern_tz.localize(dt_naive)
    dt_utc = dt_eastern.astimezone(timezone.utc)
    raw.set_meas_date(dt_utc)
    return raw

def run_prep(raw: mne.io.Raw) -> pyprep.PrepPipeline:
    """Run the PREP pipeline

    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.

    Returns:
        pyprep.PrepPipeline: The pipeline object containing all the info.
    """
    raw, montage = ch.set_channel_montage(raw)
    raw.filter(l_freq=0, h_freq=125).resample(250)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
    prep = pyprep.PrepPipeline(
        raw, 
        montage=montage, 
        channel_wise=True, 
        prep_params=prep_params
    )

    return prep.fit()

def annotate_blinks(raw: mne.io.Raw) -> mne.Annotations:
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=["Fp1", "Fp2"])
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
        orig_time=raw.info["meas_date"]
    )
    return blink_annotations

def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
        raw, 
        threshold=3, 
        ch_type='eeg', 
        min_length_good=0.1, 
        filter_freq=(95, 120),
        )

    return muscle_annotations

def read_experiment_annotations(streams_dict: dict, raw) -> mne.Annotations:

    eeg_start = datetime.fromtimestamp(streams_dict["eeg"]["time_stamps"][0])
    time_stamps = streams_dict["markers"]["time_stamps"]
    delta = [datetime.fromtimestamp(t) - eeg_start for t in time_stamps]
    onset = [d.total_seconds() for d in delta]
    duration = np.zeros_like(onset)
    description = [desc[0] for desc in streams_dict["markers"]["time_series"]]

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description,
        orig_time=raw.info["meas_date"]
    )

    return annotations

def combine_annotations(
    annotations_list: list[mne.Annotations]
                        ) -> mne.Annotations:

    return sum(annotations_list, start=mne.Annotations([],[],[]))

def save_bids_tree(raw_cleaned, bids_path):

    mne_bids.write_raw_bids(
        raw_cleaned.pick_types(eeg = True),
        bids_path=bids_path,
        allow_preload=True,
        format="EDF",
        overwrite=True,
    )

def save_eeg_coordinates(
    raw_bv,
    CapTrack
):
    pass
    
def full_pipeline(subject, 
             session,
             task, 
             run, 
             root=Path("data2/Projects/NKI_RS2/MoBI")
             ):

    base = root / f"sub-{subject}/ses-{session}/"
    lsl_filename = (
        base / f"lsl/sub-{subject}_ses-{session}_task-{task}_run-{run}_lsl.xdf.gz"
    )
    bv_filename = (
        base / f"raw/sub-{subject}_ses-{session}_run-{run}_eeg.vhdr"
    )
    try:
        raw_bv = mne.io.read_raw_brainvision(bv_filename)
    except:
        return None
    if not lsl_filename.exists():
        print(f"{lsl_filename} does not exist")
        return None

    saving_bids_path = mne_bids.BIDSPath(
        root=root / "derivatives", subject=subject, session=session, task=task, run=run
    )
    try:
        if saving_bids_path.fpath.parent.exists():
            return None
        else:
            saving_bids_path.mkdir()

    except Exception:
        saving_bids_path.mkdir()

    mne.set_log_level(verbose="ERROR")
    streams_dict = extract_streams(lsl_filename)
    raw = read_raw_xdf(streams_dict)
    
    try:
        prep_output = run_prep(raw)
        raw_cleaned = prep_output.raw
    except Exception as e:
        print(f"Error: {lsl_filename} prep:\n \t{e}")
        return None


    blinks_annotations = annotate_blinks(raw_cleaned)
    muscle_annotations = annotate_muscle(raw_cleaned)
    experiment_annotations = read_experiment_annotations(streams_dict)

    annotations = combine_annotations([
        blinks_annotations,
        muscle_annotations,
        experiment_annotations
        ])

    raw_cleaned.set_annotations(annotations)
    save_bids_tree(raw, raw_cleaned, saving_bids_path)
    ch.save_channels_info(raw_bv, prep_output, saving_bids_path)


def annotation_correction_pipeline(file: pd.Series):
    """Pipeline to do a correction of existing annotations.
    
    Args:
        file (pd.Series): Should be the tsv file of the events.

    Returns:
        None
    """

    base = file["root"].parent / f"sub-{file["subject"]}/ses-{file["session"]}/"
    lsl_filename = (
        base / f"lsl/sub-{file["subject"]}_ses-{file["session"]}_task-{file["task"]}_run-{file["run"]}_lsl.xdf.gz"
    )

    annot_csv = pd.read_csv(file["filename"], sep="\t")
    cleaned_csv = annot_csv.loc[
        (annot_csv["trial_type"] == "BAD_muscle")
        |(annot_csv["trial_type"] == "blink")
        ]

    mne.set_log_level(verbose="ERROR")
    streams_dict = extract_streams(lsl_filename)
    raw = read_raw_xdf(streams_dict)
    artifact_annot = mne.Annotations(
        onset = cleaned_csv["onset"].values,
        duration = cleaned_csv["duration"].values,
        description = cleaned_csv["trial_type"].values,
        orig_time = raw.info["meas_date"]
    )

    experiment_annotations = read_experiment_annotations(streams_dict, raw)
    raw.set_annotations(artifact_annot + experiment_annotations)
    events = mne.events_from_annotations(raw, regexp=r".*")
    annotation_df = pd.DataFrame(
        {"onset": raw.annotations.onset,
        "duration": raw.annotations.duration,
        "trial_type":raw.annotations.description,
        "value": events[0][:,2],
        "sample": events[0][:,0],
        }
    )
    annotation_df.to_csv(file["filename"], sep = "\t", index = False)
    #save_bids_tree(raw, raw_cleaned, saving_bids_path)
    #ch.save_channels_info(raw_bv, prep_output, saving_bids_path)
#%%
if __name__ == "__main__":
    root = Path("/Users/samuel/Desktop/RS_2_dev/derivatives")
    architecture = arch.BidsArchitecture(root = root)
    selection = architecture.select(
    datatype = "eeg",
    suffix = "events",
    extension = ".tsv"
)
    for idx, file in selection:
        print(f"=== PROCESSING {file['filename']} ===")
        annotation_correction_pipeline(file)
#%%
root = Path("/Users/samuel/Desktop/RS_2_dev/derivatives")
architecture = arch.BidsArchitecture(root = root)
selection = architecture.select(
datatype = "eeg",
suffix = "events",
extension = ".tsv"
)
