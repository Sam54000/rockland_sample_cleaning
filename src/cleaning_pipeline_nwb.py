#%%
"""
EEG Cleaning Pipeline for Rockland Sample Dataset

This module implements a comprehensive pipeline for preprocessing and cleaning EEG data 
from the Rockland Sample dataset. It handles the conversion from raw XDF format to 
MNE-compatible objects, applies the PREP preprocessing pipeline, detects various 
artifacts (blinks, muscle activity), and saves the cleaned data in BIDS-compliant format.

The module provides functionality for:
- Converting XDF-format EEG data to standardized voltage units
- Applying proper channel montages for EEG data
- Running the PREP pipeline for robust referencing and noise removal
- Detecting and annotating eye blinks and muscle artifacts
- Extracting and aligning experimental markers with EEG data
- Combining various annotations into a unified annotation set
- Saving processed data in BIDS-compliant format for further analysis

This pipeline is specifically designed to work with the Rockland Sample dataset
collected using the BrainVision and Lab Streaming Layer (LSL) recording systems.
"""
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

import pyxdf

def run_prep(raw: mne.io.Raw) -> pyprep.PrepPipeline:
    """Run the PREP pipeline for EEG preprocessing.
    
    Applies the PREP pipeline which includes:
    1. Setting channel montage
    2. Filtering (0-125 Hz) and resampling to 250 Hz
    3. Reference correction
    4. Line noise removal (60 Hz and harmonics)
    
    Args:
        raw (mne.io.Raw): The raw EEG data in MNE format.

    Returns:
        pyprep.PrepPipeline: The fitted PREP pipeline object containing
                             the cleaned data and preprocessing information.
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
    """Detect and annotate eye blinks in EEG data.
    
    Uses frontal electrodes (Fp1, Fp2) to detect eye blinks and
    creates annotations marking their occurrences.
    
    Args:
        raw (mne.io.Raw): The EEG data in MNE format.
        
    Returns:
        mne.Annotations: Annotations marking detected eye blinks.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=["Fp1", "Fp2"])
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
        orig_time=raw.info["meas_date"]
    )
    return blink_annotations

def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    """Detect and annotate muscle artifacts in EEG data.
    
    Uses z-score thresholding in the high-frequency band (95-120 Hz)
    to identify muscle artifacts in the EEG data.
    
    Args:
        raw (mne.io.Raw): The EEG data in MNE format.
        
    Returns:
        mne.Annotations: Annotations marking detected muscle artifacts.
    """
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
        raw, 
        threshold=3, 
        ch_type='eeg', 
        min_length_good=0.1, 
        filter_freq=(95, 120),
        )

    return muscle_annotations


def combine_annotations(
    annotations_list: list[mne.Annotations]
                        ) -> mne.Annotations:
    """Combine multiple MNE Annotations objects into a single object.
    
    Takes a list of annotation objects and combines them into a single
    annotations object. Handles empty annotation lists gracefully.
    
    Args:
        annotations_list (list[mne.Annotations]): List of annotation objects to combine.
        
    Returns:
        mne.Annotations: Combined annotations object.
    """
    return sum(annotations_list, start=mne.Annotations([],[],[]))

def save_bids_tree(raw_cleaned: mne.io.Raw, bids_path: mne_bids.BIDSPath):
    """Save preprocessed EEG data in BIDS-compliant format.
    
    Writes the cleaned EEG data to a BIDS-compliant EDF file using the
    specified BIDS path.
    
    Args:
        raw_cleaned (mne.io.Raw): Cleaned EEG data to save.
        bids_path (mne_bids.BIDSPath): BIDS path object specifying where to save the data.
        
    Returns:
        None: The function saves the EEG data to disk and doesn't return anything.
    """
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
    """Placeholder function for saving EEG electrode coordinates.
    
    This function is intended to save the coordinates of EEG electrodes,
    but is currently not implemented.
    
    Args:
        raw_bv: Raw BrainVision data.
        CapTrack: Cap tracking data.
        
    Returns:
        None
    """
    pass
    
def full_pipeline(file: dict) -> str:
    """Execute the complete EEG preprocessing pipeline for a given file.
    
    The pipeline includes:
    1. Loading BrainVision and XDF files
    2. Setting up BIDS path for saving
    3. Running PREP preprocessing
    4. Detecting and annotating artifacts (blinks, muscle)
    5. Extracting and adding experimental markers
    6. Saving the processed data in BIDS format
    
    Args:
        file (dict): Dictionary containing file information with keys:
            - 'root': Root directory path
            - 'subject': Subject ID
            - 'session': Session ID
            - 'run': Run number
            - 'task': Task name
            - 'filename': Path to the XDF file
            
    Returns:
        str: Status message indicating success ("OK"), "Already Done", or error message.
    """
    base = file["root"]/ f"sub-{file["subject"]}/ses-{file["session"]}/"
    bv_filename = (
        base / f"raw/sub-{file["subject"]}_ses-{file["session"]}_run-{file["run"]}_eeg.vhdr"
    )

    bv_alt_filename = (
        base / f"raw/sub-{file["subject"]}_ses-{file["session"]}_run-{file["run"]}.vhdr"
    )
    if not bv_filename.is_file():
        bv_filename = bv_alt_filename
        print(f"\n\n######### Using alternative filename: {bv_filename}\n\n")
    try:
        raw_bv = mne.io.read_raw_brainvision(bv_filename)
    except Exception as e:
        return str(e)
        #raise e
    saving_bids_path = mne_bids.BIDSPath(
        root=file["root"]/ "derivatives", 
        subject=file["subject"], 
        session=file["session"],
        datatype="eeg",
        task=file["task"],
        run=file["run"],
    )

    theoretical_fname = Path(os.fspath(saving_bids_path.fpath))
    
    print(f"Theoretical fname:{theoretical_fname}")
    saving_bids_path.mkdir()
    try:
        if theoretical_fname.is_file():
            return "Already Done"
        else:
            saving_bids_path.mkdir()

    except Exception as e:
        saving_bids_path.mkdir()
        return str(e)

    mne.set_log_level(verbose="ERROR")
    streams_dict = extract_streams(file["filename"])
    raw = read_raw_xdf(streams_dict)
    
    try:
        prep_output = run_prep(raw)
        raw_cleaned = prep_output.raw
    except Exception as e:
        return str(e)

    try:
        blinks_annotations = annotate_blinks(raw_cleaned)
        muscle_annotations = annotate_muscle(raw_cleaned)
        experiment_annotations = read_experiment_annotations(streams_dict, raw)

        annotations = combine_annotations([
            blinks_annotations,
            muscle_annotations,
            experiment_annotations
            ])

        raw_cleaned.set_annotations(annotations)
        save_bids_tree(raw_cleaned, saving_bids_path)
        ch.save_channels_info(raw_bv, prep_output, saving_bids_path)
    except Exception as e:
        return str(e)
    
    return "OK"

if __name__ == "__main__":
    root = Path("/data2/Projects/NKI_RS2/MoBI/")
    architecture = arch.BidsArchitecture(root = root)
    selection = architecture.select(
        datatype = "lsl", 
        suffix = "lsl",
        extension = ".gz"
)
    report = {"subject":[],
              "session":[],
              "task":[],
              "run":[],
              "message":[]
    }
    for idx, file in selection.database.iterrows():
        print(f"=== PROCESSING {file['filename']} ===")
        report["subject"].append(file["subject"])
        report["session"].append(file["session"])
        report["task"].append(file["task"])
        report["run"].append(file["run"])
        
        try:
            message = full_pipeline(file)
            report["message"].append(message)
        except Exception as e:
            report["message"].append(str(e))
            continue

    report_df = pd.DataFrame(report)
    report_df.to_csv(root / "cleaning_report_2.tsv", sep = "\t", index = False)

# %%
