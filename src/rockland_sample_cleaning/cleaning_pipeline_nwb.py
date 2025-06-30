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
import re
import mne
import pyprep
from datetime import datetime, timezone
import pytz
import mne_bids
from rockland_sample_cleaning import channels_handling as ch
from rockland_sample_cleaning import read_file
import pandas as pd
import bids_explorer.architecture as arch


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

    raw.filter(l_freq=0, h_freq=125).resample(250)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
    prep = pyprep.PrepPipeline(
        raw, 
        montage=raw.get_montage(), 
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
    
def full_pipeline(file: str, saving_bids_path: os.PathLike, overwrite = False) -> str:
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

    theoretical_fname = Path(os.fspath(saving_bids_path.fpath))
    
    print(f"Theoretical fname:{theoretical_fname}")
    saving_bids_path.mkdir()
    try:
        if theoretical_fname.is_file() and not overwrite:
            return "Already Done"
        else:
            saving_bids_path.mkdir()

    except Exception as e:
        saving_bids_path.mkdir()
        return str(e)

    mne.set_log_level(verbose="ERROR")
    raw, channels = read_file.read_raw_nwb(file)
    
    try:
        prep_output = run_prep(raw)
        raw_cleaned = prep_output.raw
    except Exception as e:
        return str(e)

    try:
        blinks_annotations = annotate_blinks(raw_cleaned)
        muscle_annotations = annotate_muscle(raw_cleaned)

        annotations = combine_annotations([
            blinks_annotations,
            muscle_annotations,
            raw.annotations,
            ])

        raw_cleaned.set_annotations(annotations)
        save_bids_tree(raw_cleaned, saving_bids_path)
        ch.save_nwb_channels_info(channels, prep_output, saving_bids_path)
    except Exception as e:
        return str(e)
    
    return "OK"

if __name__ == "__main__":
    root = Path("/data2/Projects/NKI_RS2/MoBI/NWB_BIDS")
    report = {"subject":[],
              "session":[],
              "task":[],
              "run":[],
              "message":[]
    }
    for file in root.rglob(r"*"):

        if not file.suffix == ".nwb":
            continue
        subject = re.findall(r"sub-.*?(?=_)",file.name)[0].split("-")[-1]
        session = re.findall(r"ses-.*?(?=_)",file.name)[0].split("-")[-1]
        task = re.findall(r"task-.*?(?=_)",file.name)[0].split("-")[-1]
        run = re.findall(r"run-.*?(?=_)",file.name)[0].split("-")[-1]
        print(f"=== PROCESSING {file.name} ===")
        report["subject"].append(subject)
        report["session"].append(session)
        report["task"].append(task)
        report["run"].append(run)
        
        try:
            saving_bids_path = mne_bids.BIDSPath(
                root=root.parent / "derivatives", 
                subject=subject,
                session=session,
                datatype="eeg",
                task=task,
                run=run,
            )
            message = full_pipeline(file, saving_bids_path, overwrite = True)
            report["message"].append(message)

        except Exception as e:
            report["message"].append(str(e))
            continue

    report_df = pd.DataFrame(report)
    report_df.to_csv(root.parent / "derivatives" / "cleaning_report_nwb.tsv", sep = "\t", index = False)

# %%
