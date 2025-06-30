import os
import mne
import numpy as np
import pandas as pd

def find_real_channel_name(raw: mne.io.Raw, name: str = "ecg") -> list:
    """Find the name as it is in the raw object.

    Channel names vary across different EEG systems and manufacturers. It varies
    in terms of capitalization, spacing, and special characters. This function
    finds the real name of the channel in the raw object.

    Args:
        raw (mne.io.Raw): The mne Raw object
        name (str): The name of the channel to find in lower case.

    Returns:
        str: The real name of the channel in the raw object.
    """
    channel_found = list()
    for ch_name in raw.info["ch_names"]:
        if name.lower() in ch_name.lower():
            channel_found.append(ch_name)
    return channel_found

def map_channel_type(raw: mne.io.Raw) -> dict:
    """Find and map into MNE type the ECG and EOG channels.

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        dict: dictionary of channel type to map into `raw.set_channel_types` method
    """
    channels_map = dict()
    for ch_type in ["ecg", "eog"]:
        ch_name_in_raw = find_real_channel_name(raw, ch_type)
        if ch_name_in_raw:
            if len(ch_name_in_raw) == 1:
                channels_map.update({ch_name_in_raw[0]: ch_type})
            elif len(ch_name_in_raw) > 1:
                for name in ch_name_in_raw:
                    channels_map.update({name: ch_type})
        else:
            print(f"No {ch_type.upper()} channel found.")
            if ch_type == "eog":
                print("Fp1 and Fp2 will be used for EOG signal detection")

    return channels_map

def all_conditions_met_on(chan: dict) -> bool:
    """Check if a channel has all required fields populated.
    
    Verifies that both 'type' and 'label' fields exist and are non-empty
    in the channel dictionary from the XDF data object.

    Args:
        chan (dict): A dictionary from the xdf data object. Usually accessed by
                     `eeg_stream['info']['desc'][0]['channels'][0]['channel']`

    Returns:
        bool: True if both 'type' and 'label' fields exist and are non-empty,
              False otherwise or if an exception occurs
    """
    try:
        return chan.get("type",[]) != [] and chan.get("label", []) != []
    except Exception:
        return False

def parse_lsl_channel_names(lsl_eeg_stream: dict) -> dict:
    """Parse the LSL channels into MNE-compatible format.
    
    From the XDF stream dedicated to the EEG, this function generates a dictionary
    containing the name and type of each channel in a format compatible with MNE.
    Special handling is done for marker channels, which are converted to 'stim' type
    as required by MNE.

    Args:
        lsl_eeg_stream (dict): The XDF stream dedicated to EEG data.

    Returns:
        dict: A dictionary with two keys:
            - 'ch_names': List of channel names
            - 'ch_types': List of channel types in MNE format
              (with 'marker' type converted to 'stim')
    """
    chan_dict = lsl_eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = list()
    ch_types = list()
    for chan in chan_dict:
        if not all_conditions_met_on(chan):
            continue

        ch_names.append(chan["label"][0])
        if chan["type"][0].lower() == "marker":
            ch_types.append("stim")
        else:
            ch_types.append(chan["type"][0].lower())

    parsed_chan_info = {
        "ch_names": ch_names,
        "ch_types": ch_types,
    }

    return parsed_chan_info

def set_channel_montage(raw_bv) -> tuple:
    """Set the electrode montage for the EEG data.
    
    Creates a standard 'easycap-M1' montage, maps channel types,
    and applies the montage to the raw EEG data.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data.

    Returns:
        tuple: A tuple containing:
            - raw_bv (mne.io.Raw): The modified raw object with montage applied
            - montage (mne.channels.montage): The applied electrode montage
    """
    montage = mne.channels.make_standard_montage("easycap-M1")
    channel_map = map_channel_type(raw_bv)
    raw_bv.set_channel_types(channel_map)
    raw_bv.set_montage(montage, on_missing="warn")
    return raw_bv, montage

def set_channel_dataframe(raw_bv, prep_output) -> pd.DataFrame:
    """Create a DataFrame with channel information and quality metrics.
    
    Constructs a DataFrame that contains channel names, flags for different
    types of noisy channels, and impedance values.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data and impedance values.
        prep_output: Object containing preprocessing results, with attributes:
            - noisy_channels_original (dict): Dictionary mapping noise labels to channel lists
            - still_noisy_channels (list): List of channels that remain noisy after preprocessing

    Returns:
        pd.DataFrame: DataFrame with channel names as index and columns for:
            - Various noise flags from prep_output.noisy_channels_original
            - 'still_noisy' flag for channels that remain noisy
            - 'impedances' values for each channel
    """
    df_dict = {"name": raw_bv.impedances.keys()}
    for bad_label, bad_ch in prep_output.noisy_channels_original.items():
        df_dict[bad_label] = np.isin(df_dict["name"], bad_ch)
    df_dict["still_noisy"] = np.isin(df_dict["name"], prep_output.still_noisy_channels)
    df_dict["impedances"] = [
        chi["imp"] for chn, chi in raw_bv.impedances.items() if chn.lower()
    ]

    return pd.DataFrame(df_dict).set_index("name")

def set_nwb_channel_dataframe(
    channels: pd.DataFrame, 
    prep_output
    ) -> pd.DataFrame:
    """Create a DataFrame with channel information and quality metrics.
    
    Constructs a DataFrame that contains channel names, flags for different
    types of noisy channels, and impedance values.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data and impedance values.
        prep_output: Object containing preprocessing results, with attributes:
            - noisy_channels_original (dict): Dictionary mapping noise labels to channel lists
            - still_noisy_channels (list): List of channels that remain noisy after preprocessing

    Returns:
        pd.DataFrame: DataFrame with channel names as index and columns for:
            - Various noise flags from prep_output.noisy_channels_original
            - 'still_noisy' flag for channels that remain noisy
            - 'impedances' values for each channel
    """
    df_dict = {"name": [s.split(" ")[-1] 
                        for s in channels["group_name"].values]}
    for bad_label, bad_ch in prep_output.noisy_channels_original.items():
        if bad_label != "bad_all":
            df_dict[bad_label] = np.isin(df_dict["name"], bad_ch)
    df_dict["still_noisy"] = np.isin(df_dict["name"], prep_output.still_noisy_channels)
    df_dict["impedances"] = channels["allImpedances"].values

    return pd.DataFrame(df_dict).set_index("name")


def save_channels_info(raw_bv, prep_output, saving_bids_path):
    """Save channel information to a BIDS-compliant TSV file.
    
    Creates a channel information DataFrame using set_channel_dataframe(),
    then merges it with any existing channel information and saves it
    to a TSV file following BIDS formatting conventions.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data.
        prep_output: Object containing preprocessing results with noisy channel information.
        saving_bids_path: Object with attributes:
            - fpath (pathlib.Path): Path object for the parent directory
            - basename (str): Base filename to use for the output file

    Returns:
        None: The function saves the channel information to a TSV file and doesn't return anything.
    """
    channel_dataframe = set_channel_dataframe(raw_bv, prep_output)

    channel_info_fname = saving_bids_path.fpath.parent / (
        os.fspath(saving_bids_path.basename) + "_channels.tsv"
    )

    channel_info_dataframe = pd.read_csv(
        channel_info_fname, sep="\t", index_col=["name"]
    )

    result = channel_info_dataframe.join(channel_dataframe, how="outer")
    result.to_csv(channel_info_fname, sep="\t")

def save_nwb_channels_info(
    channels: pd.DataFrame,
    prep_output,
    saving_bids_path
    ):
    """Save channel information to a BIDS-compliant TSV file.
    
    Creates a channel information DataFrame using set_channel_dataframe(),
    then merges it with any existing channel information and saves it
    to a TSV file following BIDS formatting conventions.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data.
        prep_output: Object containing preprocessing results with noisy channel information.
        saving_bids_path: Object with attributes:
            - fpath (pathlib.Path): Path object for the parent directory
            - basename (str): Base filename to use for the output file

    Returns:
        None: The function saves the channel information to a TSV file and doesn't return anything.
    """
    channel_dataframe = set_nwb_channel_dataframe(channels, prep_output)

    

    channel_info_fname = saving_bids_path.fpath.parent / (
        os.fspath(saving_bids_path.basename) + "_channels.tsv"
    )

    channel_info_dataframe = pd.read_csv(
        channel_info_fname, sep="\t", index_col=["name"]
    )

    result = channel_info_dataframe.join(channel_dataframe, how="outer")
    values_to_replace = result.loc[~result["type"].isna().values]
    columns_to_fill = [
        "type",
        "units",
        "low_cutoff",
        "high_cutoff",
        "description",
        "sampling_frequency",
    ]
    for col in columns_to_fill:
        result.loc[result[col].isna().values,col] = values_to_replace[col].iloc[0]

    result.loc[result["still_noisy"], "status"] = "bad"
    result.to_csv(channel_info_fname, sep="\t")