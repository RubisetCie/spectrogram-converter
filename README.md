## wav2png

Run this script to convert wav files to spectrograms, which are saved as png files.
Is able to run on a folder structure with class labels:
      
      root/dog/0001.wav
      root/dog/0002.wav

      root/cat/0001.wav
      root/cat/0002.wav   etc.
Or otherwise on single files.

Example use:  

**For class folder structure**
```bash
python ./wav2png.py folder --rootdir [rootdir]
```
**For single files**
```bash
 python ./wav2png.py single --filename [filename.wav]
```
Scaling is done on the STFT output to be compatible with 8-bit png format. The script searches the dataset for the maximum and minimum values, rounds up and down to the nearest integer respectively then scales to [0,255].

## png2wav

Run to convert individual png spectrograms back to wav. Script assumes (and inverts) similar scaling as in wav2png. Griffin-Lim algortihm is initialized with [SPSI](http://ieeexplore.ieee.org/abstract/document/7251907/). SPSI code originally from [here](https://github.com/lonce/SPSI_Python).

Example use:  

**For single png spectrogram**
```bash
python ./png2wav.py [filename.png]
```
