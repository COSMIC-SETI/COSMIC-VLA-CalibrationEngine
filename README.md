# COSMIC-VLA-CalibrationEngine

This repository is minimal. It primarily houses the `calibrate_uvh5.py` script which is used to extract gains from an input *.uvh5 file. A detailed description of how this script fits into the overall calibration system is described in this [readme](https://github.com/COSMIC-SETI/COSMIC-VLA-DelayEngine).

`calibrate_uvh5.py` take arguments:
- `-d` : the path to the input *.uvh5 file.
- `-o` : the output directory to which to save the residual delays / phases / gains json dictionary.
- `-ad` : to which *.uvh5 to apply the calculated solutions.
- `--gengain` : if set, generate a json file of output gain per antenna/freq/pol.
- `--gendelay` : if set, generate a file of output delays per antpol.
- `--genphase` : if set, generate a file of output phases per antpol.
- `--pub-to-redis` : set up a redis object, and publish the residual delays/phases/gains to hashes in it.

In production, `calibrate_uvh5.py` is run by the [postprocessor](https://github.com/COSMIC-SETI/COSMIC-VLA-PythonLibs/blob/main/scripts/postprocess_hub.py) to generate `gains` off of a recently recorded calibration observation *.uvh5 file and publish the results to Redis like so:
```python calibrate_uvh5.py -d path/to/uvh5file -o path/to/where/gains/are/logged --gengain --pub-to-redis```

This script contains code to produce diagnostic plots from the ingested uvh5 file but at present these are not available through arguments (TODO).

sliding_rfi_flagger: Given a spectra, the code utilizes a sliding median window which move across the data to create a smooth bandpass model. The smooth bandpass model will subtracted from the data to search for RFI signals greater than a threshold. The code works best to remove narrowband RFI signals. The sliding window size and threshold detection can be modified. This is used to avoid bad RFI channels while calibrating the data.
