# TOF analysis
This script reads the waveforms stored in 'dt5743wave0i' (i=0-7) branches in 'waveform_tree' generated by midas2root converter. 

## Run time 
Move into the TOF_analysis repository, then 
```python TOF_analysis_8channels.py <path_to_output.root> <run_number>```
Make sure that ```python >3.0```. You may have to use ```python3``` instead of ```python``` on Ubuntu. 

## Dependencies
Packages: Python >3, uproot4, numpy and matplotlib.
Input file: root file generated by a [modified midas2root converter](https://github.com/luanviko/midas2root_mppc_simpler).

To install pip in your system, use ```python -m ensurepip --upgrade```. 

To install numpy, use: ```pip3 install -U numpy```. Do the same for ```uproot``` and ```matplotlib```.



