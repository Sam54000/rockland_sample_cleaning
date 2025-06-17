# Rockland Sample EEG cleaning
This code is to clean the EEG of the Rockland Sample. It works but it still
under development.

# Installation
To use the codes you will need to install it following a number of steps.
It is strongly recommended that you use a virtual environment and package 
manager. I personally use [Anaconda](https://www.anaconda.com/download).


## Required packages
1. [MNE](https://mne.tools/stable/install/manual_install.html#manual-install) (if it is not already done)
2. [poetry](https://python-poetry.org/docs/#installation)
When following their instruction you will create a virtual environment named `mne` (or the name you chose)
3. In the same virtual environment install [bids_explorer](https://pypi.org/project/bids_explorer/)

## Install this codebase with Poetry
1. Make sure you activate your virtual environment (in my case using Anaconda I want to activate my `mne` environment:
`conda activate mne`)
2. In your terminal make sure you are within the desired directory (`cd <myprojects>` 
replace `<myproject>` by the name of the folder you created for this specific project)
3. 
4. 
5. Go to the cloned repository `cd rockland_sample_cleaning`
6. Run `poetry install`


References
----------
Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896).https://doi.org/10.21105/joss.01896

Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103.https://doi.org/10.1038/s41597-019-0104-8

