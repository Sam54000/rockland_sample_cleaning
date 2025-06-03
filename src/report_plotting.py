#%%
"""
EEG Quality and Impedance Reporting for Rockland Sample Dataset

This module provides visualization and reporting tools to assess the quality of EEG recordings
from the Rockland Sample dataset. It creates comprehensive reports on electrode impedances,
data quality metrics, and power spectral density changes before and after cleaning.

The module offers functionality for:
- Loading and processing impedance data from TSV files
- Computing average impedances across recordings
- Visualizing electrode impedances on topographic maps
- Plotting the distribution of bad channel detection across recordings
- Calculating and comparing power spectral density before and after cleaning
- Generating multi-page PDF reports summarizing data quality metrics
- Extracting and converting XDF stream data for analysis

The visualizations help researchers evaluate the effectiveness of the cleaning pipeline
and identify patterns in data quality issues across different recording sites and sessions.
"""
import matplotlib.pyplot as plt
import mne
import pandas as pd
import pytz
from datetime import datetime, timezone
import pyxdf
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import channels_handling as ch
from pathlib import Path
from mne.viz.utils import _plot_sensors_2d
from bids_explorer.architecture.architecture import BidsArchitecture
import seaborn as sns

def load_impedances_from_tsv(tsv_path):
    """Load electrode impedance data from a TSV file.
    
    Extracts electrode names and their corresponding impedance values,
    filtering out non-electrode channels like GND and REF.
    
    Args:
        tsv_path (str or Path): Path to the TSV file containing impedance data.
        
    Returns:
        pd.Series: Series with electrode names as index and impedance values in kΩ.
    """
    data = pd.read_csv(tsv_path, sep='\t')
    impedance_col_name = [col for col in data.columns if "impedance" in col.lower()][0]
    impedance_data = data[['name', impedance_col_name]].set_index('name').dropna()
    impedance_data = impedance_data[~impedance_data.index.str.lower().isin(["gnd", "ref"])]
    return impedance_data[impedance_col_name].astype(float)

def compute_average_impedance(tsv_paths):
    """Compute average impedance values across multiple recordings.
    
    Loads impedance data from multiple TSV files and calculates the
    mean impedance for each electrode across all recordings.
    
    Args:
        tsv_paths (list): List of paths to TSV files containing impedance data.
        
    Returns:
        pd.Series: Series with electrode names as index and average impedance values.
    """
    impedances_list = [load_impedances_from_tsv(path) for path in tsv_paths]
    combined_impedances = pd.concat(impedances_list, axis=1)
    avg_impedances = combined_impedances.mean(axis=1, skipna=True)
    return avg_impedances

def plot_average_impedance(avg_impedances, montage_name='easycap-M1', vmin=None, vmax=None, title='Average Impedance', show=True):
    """Plot average impedance values on a topographic EEG electrode map.
    
    Creates a color-coded topographic map showing the average impedance
    values for each electrode using a standard montage.
    
    Args:
        avg_impedances (pd.Series): Series with electrode names as index and impedance values.
        montage_name (str): Name of the standard electrode montage to use.
        vmin (float, optional): Minimum value for color scaling. Defaults to minimum value in data.
        vmax (float, optional): Maximum value for color scaling. Defaults to maximum value in data.
        title (str): Title for the plot.
        show (bool): Whether to display the plot immediately.
        
    Returns:
        tuple: (figure, axes) matplotlib objects for further customization.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = montage.ch_names

    # Filter montage positions for available impedances
    valid_ch_names = [ch for ch in ch_names if ch in avg_impedances.index]
    positions = np.array([montage.get_positions()['ch_pos'][ch][:2] for ch in valid_ch_names])

    impedances = avg_impedances.loc[valid_ch_names].values

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=vmin or np.nanmin(impedances), vmax=vmax or np.nanmax(impedances))
    colors = cmap(norm(impedances))

    fig, axes = plt.subplots(1, 2, figsize=(7, 6), gridspec_kw={'width_ratios': [1, 0.05]})

    fake_info = mne.create_info(ch_names=valid_ch_names, sfreq=1, ch_types='eeg')
    fake_info.set_montage(montage)

    figure, axes = plt.subplots(1,2,figsize=(5,5),
                                width_ratios=[1,0.05],
                                )
    _plot_sensors_2d(
        positions,
        fake_info,
        to_sphere = True,
        picks=None,
        colors=colors,
        bads=[],
        ch_names=valid_ch_names,
        title=title,
        show_names=True,
        ax=axes[0],
        show=False,
        kind='topomap',
        block=False,
        sphere=None,
        pointsize=50,
        linewidth=0
    )

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[1])
    cb.set_label('Average Impedance (kΩ)')
    axes[0].set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return figure, axes

BOOLEAN_COLUMNS = [
    "bad_by_nan", "bad_by_flat", "bad_by_deviation", "bad_by_hf_noise",
    "bad_by_correlation", "bad_by_SNR", "bad_by_dropout", "bad_by_ransac"
]

def load_boolean_counts(tsv_paths, columns):
    """Load and count boolean flags for bad channels across multiple recordings.
    
    For each type of bad channel detection method, counts how many times
    each electrode was flagged as bad across all recordings.
    
    Args:
        tsv_paths (list): List of paths to TSV files containing channel quality data.
        columns (list): List of column names for boolean flags to count.
        
    Returns:
        dict: Dictionary with column names as keys and Series of counts as values.
    """
    bool_counts = {col: pd.Series(dtype=int) for col in columns}

    for path in tsv_paths:
        df = pd.read_csv(path, sep='\t').set_index('name')
        for col in columns:
            if col in df:
                bool_series = df[col].fillna(False).astype(bool).astype(int)
                bool_counts[col] = bool_counts[col].add(bool_series, fill_value=0)

    # Convert counts to integers
    bool_counts = {col: counts.astype(int) for col, counts in bool_counts.items()}
    return bool_counts

def plot_boolean_counts(bool_counts, montage_name='easycap-M1', title=None, show=True):
    """Plot topographic maps of bad channel counts for each detection method.
    
    Creates a grid of topographic maps showing how many times each electrode
    was flagged as bad by different detection methods across recordings.
    
    Args:
        bool_counts (dict): Dictionary with column names as keys and Series of counts as values.
        montage_name (str): Name of the standard electrode montage to use.
        title (str, optional): Title for the entire figure.
        show (bool): Whether to display the plot immediately.
        
    Returns:
        tuple: (figure, axes) matplotlib objects for further customization.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos_dict = montage.get_positions()['ch_pos']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    cmap = plt.get_cmap('Reds')

    for ax, (col, counts) in zip(axes, bool_counts.items()):
        valid_ch_names = [ch for ch in montage.ch_names if ch in counts.index]
        positions = np.array([ch_pos_dict[ch][:2] for ch in valid_ch_names])
        counts_values = counts.loc[valid_ch_names].values
        
        # Set vmax to 1 if maximum is 0
        vmax = np.max(counts_values)
        if vmax == 0:
            vmax = 1
            
        norm = plt.Normalize(vmin=0, vmax=vmax)

        colors = cmap(norm(counts_values))

        fake_info = mne.create_info(ch_names=valid_ch_names, sfreq=1, ch_types='eeg')
        fake_info.set_montage(montage)

        _plot_sensors_2d(
            positions,
            fake_info,
            picks=None,
            colors=colors,
            bads=[],
            ch_names=valid_ch_names,
            title=f'Count of {col}',
            show_names=True,
            ax=ax,
            show=False,
            kind='topomap',
            block=False,
            to_sphere = True,
            sphere=None,
            pointsize=50,
            linewidth=0
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.set_title(col)
        plt.suptitle(title)
        
        # Create colorbar with integer ticks
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Number of recordings')
        
        # Set integer ticks
        import matplotlib.ticker as ticker
        tick_locator = ticker.MaxNLocator(integer=True)
        cbar.locator = tick_locator
        cbar.update_ticks()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
#%%
# %%
def convert_in_volts(eeg_stream: dict) -> np.ndarray:
    """Convert the EEG signal from the XDF object to volts.
    
    Extracts unit information from the XDF object and applies appropriate
    conversion factors to standardize the EEG signal in volts.

    Args:
        eeg_stream (dict): The stream in the XDF object dedicated to the EEG.

    Returns:
        np.ndarray: The EEG signal converted to volts.
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
    """Extract EEG and marker streams from an XDF file.
    
    Loads and separates the EEG data and stimulus marker streams from the XDF file.
    
    Args:
        filename (str or Path): Path to the XDF file to extract streams from.
        
    Returns:
        dict: Dictionary containing two keys:
            - 'eeg': The EEG data stream
            - 'markers': The stimulus markers stream
    """
    streams_dict = {
        "eeg": pyxdf.load_xdf(filename, select_streams=[{"type":"EEG"}])[0][0],
        "markers": pyxdf.load_xdf(filename, select_streams=[{"name":"StimLabels"}])[0][0]
    }
    return streams_dict

def read_raw_xdf(streams_dict: dict) -> mne.io.RawArray:
    """Convert XDF streams into an MNE RawArray object.
    
    Processes the XDF streams, converts EEG signals to volts, creates appropriate
    channel info, and sets the measurement date in UTC using the timestamps
    from the EEG data (converting from Eastern time).

    Args:
        streams_dict (dict): Dictionary containing 'eeg' and 'markers' streams.

    Returns:
        mne.io.RawArray: MNE raw object with properly formatted EEG data.
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

def calculate_psd(raw, fmin=1, fmax=50, n_overlap=0, n_jobs=-1):
    """Calculate power spectral density (PSD) from raw EEG data.
    
    Computes the PSD for all EEG channels using Welch's method and
    converts the power values to decibels.
    
    Args:
        raw (mne.io.Raw): MNE Raw object containing EEG data
        fmin (float): Minimum frequency to include in the PSD
        fmax (float): Maximum frequency to include in the PSD
        n_overlap (int): Number of points to overlap between segments
        n_jobs (int): Number of jobs for parallel processing
        
    Returns:
        tuple: (frequencies, psd_data_in_db) where:
            - frequencies is an array of frequency points
            - psd_data_in_db is the power spectral density in dB
    """
    # Pick only EEG channels
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    
    # Calculate PSD using Welch's method
    raw.filter(l_freq=0, h_freq=100)
    if raw.info["sfreq"] > 250:
        raw.resample(250)
    
    psds = raw.compute_psd(
        fmin=fmin,
        fmax=fmax, 
        n_overlap=n_overlap, 
        n_jobs=n_jobs, 
        picks=picks
        )
    
    # Convert power to dB
    psds_db = 10 * np.log10(psds.get_data())
    
    return psds.freqs, psds_db

def plot_single_comparison_psd(before_raw, after_raw, fmin=1, fmax=50, title=None):
    """Plot comparison of PSD before and after cleaning for a single recording.
    
    Creates a plot showing the mean and standard deviation of power spectral density
    across channels before and after cleaning, highlighting the effects of the
    cleaning pipeline.
    
    Args:
        before_raw (mne.io.Raw): Raw EEG data before cleaning
        after_raw (mne.io.Raw): Raw EEG data after cleaning
        fmin (float): Minimum frequency to include in the PSD
        fmax (float): Maximum frequency to include in the PSD
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Calculate PSDs
    freqs_before, psds_before = calculate_psd(before_raw, fmin=fmin, fmax=fmax)
    freqs_after, psds_after = calculate_psd(after_raw, fmin=fmin, fmax=fmax)
    
    # Calculate mean and std across channels
    mean_before = np.mean(psds_before, axis=0)
    std_before = np.std(psds_before, axis=0)
    
    mean_after = np.mean(psds_after, axis=0)
    std_after = np.std(psds_after, axis=0)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot before cleaning
    plt.fill_between(
        freqs_before, 
        mean_before - std_before, 
        mean_before + std_before, 
        alpha=0.2, 
        color='red'
    )
    plt.plot(freqs_before, mean_before, color='red', linewidth=2, label='Before cleaning')
    
    # Plot after cleaning
    plt.fill_between(
        freqs_after, 
        mean_after - std_after, 
        mean_after + std_after, 
        alpha=0.2, 
        color='blue'
    )
    plt.plot(freqs_after, mean_after, color='blue', linewidth=2, label='After cleaning')
    
    # Set plot labels and properties
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title(title or 'PSD Comparison: Before vs After Cleaning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multi_comparison_psd(
    before_after_pairs, 
    fmin=1,
    fmax=50, 
    title=None, 
    ax=None,
    ):
    """Plot average PSD comparison across multiple recordings.
    
    Creates a plot showing the mean and standard deviation of power spectral density
    across multiple recordings before and after cleaning, providing a more
    robust evaluation of the cleaning pipeline's effects.
    
    Args:
        before_after_pairs (list): List of tuples (before_raw, after_raw) for multiple recordings
        fmin (float): Minimum frequency to include in the PSD
        fmax (float): Maximum frequency to include in the PSD
        title (str, optional): Plot title
        ax (matplotlib.axes.Axes, optional): Axes object to plot on
        
    Returns:
        tuple: (figure, axes) matplotlib objects for further customization
    """
    all_before_psds = []
    all_after_psds = []
    freqs = None
    
    # Process each recording pair
    for before_raw, after_raw in before_after_pairs:
        # Calculate PSDs
        freqs_before, psds_before = calculate_psd(before_raw, fmin=fmin, fmax=fmax)
        freqs_after, psds_after = calculate_psd(after_raw, fmin=fmin, fmax=fmax)
        
        # Store frequency for later use
        if freqs is None:
            freqs = freqs_before
            
        # Store mean PSD across channels for each recording
        all_before_psds.append(np.mean(psds_before, axis=0))
        all_after_psds.append(np.mean(psds_after, axis=0))
    
    # Convert to arrays
    all_before_psds = np.array(all_before_psds)
    all_after_psds = np.array(all_after_psds)
    
    # Calculate mean and std across recordings
    mean_before = np.mean(all_before_psds, axis=0)
    std_before = np.std(all_before_psds, axis=0)
    
    mean_after = np.mean(all_after_psds, axis=0)
    std_after = np.std(all_after_psds, axis=0)
    
    # Plot before cleaning
    ax.fill_between(
        freqs_before, 
        mean_before - std_before, 
        mean_before + std_before, 
        alpha=0.2, 
        color='red'
    )
    ax.plot(freqs_before, mean_before, 
            color='red', 
            linewidth=2, 
            label='Before cleaning',
            )
    
    # Plot after cleaning
    ax.fill_between(
        freqs_after, 
        mean_after - std_after, 
        mean_after + std_after, 
        alpha=0.2, 
        color='blue'
    )
    ax.plot(freqs_after, mean_after, color='blue', linewidth=2, label='After cleaning')
    
    # Set plot labels and properties
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB)')
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    return ax

def get_before_after_pairs(bids_db, num_recordings=5, random_seed=None):
    """Get pairs of EEG recordings before and after cleaning.
    
    Randomly samples recordings from the BIDS database and loads both the
    original XDF data and the cleaned EDF data for comparison.
    
    Args:
        bids_db (pd.DataFrame): BIDS database from BidsArchitecture
        num_recordings (int): Number of recordings to sample
        random_seed (int, optional): Random seed for reproducible sampling
        
    Returns:
        list: List of tuples (before_raw, after_raw) containing pairs of
              MNE Raw objects for each recording before and after cleaning
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Sample random indices from database
    random_indices = np.random.choice(bids_db.index, size=min(num_recordings, len(bids_db)), replace=False)
    
    pairs = []
    max_times = []
    for idx in random_indices:
        # Get cleaned data
        after_cleaning = mne.io.read_raw(bids_db.loc[idx]["filename"], preload=True)
        
        subject = f"sub-{bids_db.loc[idx]['subject']}"
        session = f"ses-{bids_db.loc[idx]['session']}"
        task = f"task-{bids_db.loc[idx]['task']}"
        run = f"run-{bids_db.loc[idx]['run']}"

        before_cleaning_path = bids_db.loc[idx]["root"].parent / subject / session / "lsl" / f"{subject}_{session}_{task}_{run}_lsl.xdf.gz"
        before_streams = extract_streams(before_cleaning_path)
        before_cleaning_raw = read_raw_xdf(before_streams)
        max_times.append(before_cleaning_raw.times[-1])
        pairs.append((before_cleaning_raw, after_cleaning))
    
    min_time = np.floor(min(max_times))
    for pair in pairs:
        pair[0].crop(tmin = 0, tmax = min_time)
        pair[1].crop(tmin = 0, tmax = min_time)
        
    return pairs

def create_combined_figure(pairs, tsv_files, task, num_recordings):
    """Create a combined figure with PSD comparison and average impedance plots.
    
    Args:
        pairs (list): List of (before_raw, after_raw) tuples for PSD comparison
        tsv_files (list): List of paths to TSV files containing impedance data
        task (str): Task name for plot titles
        num_recordings (int): Number of recordings used for PSD comparison
        
    Returns:
        matplotlib.figure.Figure: Combined figure with both plots
    """
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
    
    # Left subplot - PSD comparison
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Calculate PSDs for all pairs
    all_before_psds = []
    all_after_psds = []
    freqs = None
    
    plot_multi_comparison_psd(pairs, fmin=1, fmax=50, title=f"PSD Comparison: {task}", ax=ax1)
    
    # Right side - Impedance plot
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Calculate average impedances
    average_impedances = compute_average_impedance(tsv_files)
    
    # Get montage and positions for impedance plot
    montage = mne.channels.make_standard_montage('easycap-M1')
    ch_names = montage.ch_names

    # Filter montage positions for available impedances
    valid_ch_names = [ch for ch in ch_names if ch in average_impedances.index]
    positions = np.array([montage.get_positions()['ch_pos'][ch][:2] for ch in valid_ch_names])

    impedances = average_impedances.loc[valid_ch_names].values

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=50)
    colors = cmap(norm(impedances))

    fake_info = mne.create_info(ch_names=valid_ch_names, sfreq=1, ch_types='eeg')
    fake_info.set_montage(montage)

    _plot_sensors_2d(
        positions,
        fake_info,
        to_sphere=True,
        picks=None,
        colors=colors,
        bads=[],
        ch_names=valid_ch_names,
        title=f"Average Impedance: {task}",
        show_names=True,
        ax=ax2,
        show=False,
        kind='topomap',
        block=False,
        sphere=None,
        pointsize=100,
        linewidth=0
    )

    # Add colorbar for impedance plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('Average Impedance (kΩ)')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":

    root = Path("/data2/Projects/NKI_RS2/MoBI/derivatives")
    architecture = BidsArchitecture(
        root = root,
        )
    num_recordings = 5
    saving_path = root / "figures"
    saving_path.mkdir(parents=True, exist_ok=True)
    with PdfPages(saving_path / "report.pdf") as pdf:
        for task in architecture.tasks:
            selection = architecture.select(task = task,
                                            suffix = "eeg",
                                            extension = ".edf")
            pairs = get_before_after_pairs(selection.database, 
                                        num_recordings=num_recordings, 
                                        random_seed=42)

            sub_saving_path = saving_path / task
            sub_saving_path.mkdir(parents=True, exist_ok=True)
            
            channels_selection = architecture.select(
                task = task,
                suffix = "channels",
                extension = ".tsv",
                )

            tsv_files = channels_selection.database["filename"].values
            
            # Create and save the combined figure
            combined_fig = create_combined_figure(pairs, tsv_files, task, num_recordings)
            combined_fig.savefig(sub_saving_path / f"Combined_PSD_Impedance_{task}.png")
            pdf.savefig(combined_fig)
            boolean_counts = load_boolean_counts(tsv_files, BOOLEAN_COLUMNS)
            fig, ax = plot_boolean_counts(
                boolean_counts, 
                montage_name='easycap-M1',
                title=f"Amount of bad channels: {task}",
                show=False
                )
            fig.savefig(sub_saving_path / f"Amount_of_bad_channels_{task}.png")
            pdf.savefig(fig)
        pdf.close()
# %%
