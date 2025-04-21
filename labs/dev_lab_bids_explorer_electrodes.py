#%%
import bids_explorer
import bids_explorer.architecture
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from bids_explorer.utils import parsing
@dataclass
class Electrodes:
    def from_file(self, electrode_file: Path, channels_file: Path):
        electrode_data = pd.read_csv(electrode_file, sep="\t")
        channels_data = pd.read_csv(channels_file, sep="\t")
        entities = parsing.parse_bids_filename(electrode_file)
        entities.update(parsing.parse_bids_filename(channels_file))

        self.data = electrode_data.merge(channels_data, on="name", how="outer")

        self.spaces = [entities["space"]]
        self.subjects = [entities["subject"]]
        self.sessions = [entities["session"]]
        self.datatypes = [entities["datatype"]]
        self.tasks = [entities.get("task", None)]
        self.runs = [entities.get("run", None)]
        self.acquisitions = [entities.get("acquisition", None)]
        self.descriptions = [entities.get("description", None)]

        return self

    def from_dataframe(self, dataframe=pd.DataFrame):
        self.data = dataframe
        self.spaces = dataframe["space"].unique()
        self.subjects = dataframe["subject"].unique()
        self.sessions = dataframe["session"].unique()
        self.datatypes = dataframe["datatype"].unique()
        self.tasks = dataframe["task"].unique()
        self.runs = dataframe["run"].unique()
        self.acquisitions = dataframe["acquisition"].unique()
        self.descriptions = dataframe["description"].unique()
#%%
root = Path("/data2/Projects/NKI_RS2/MoBI/derivatives")
elec_architecture = bids_explorer.architecture.ElectrodesArchitecture(
    root=root,
    subject = "M10983486",
    session = "MOBI2C",
    task = "checkerboard",
    run = "01"
)
# %%
elec = Electrodes()
elec.from_file(elec_architecture._electrodes_database['filename'].values[0],
                              elec_architecture._channel_database['filename'].values[0])
#%%

    