# Spectrogram Converter

These scripts are able to convert WAV files to spectrograms, which are saved as png files.

Is able to run on a folder structure with class labels:

	root/dog/0001.wav
	root/dog/0002.wav

	root/cat/0001.wav
	root/cat/0002.wav
	... etc.

Or otherwise on single files.

## Requirements:

- soundfile
- scipy
- numpy
- pillow
- librosa

## Examples:

**For class folder structure:**

```bash
python3 ./wav2png.py folder --rootdir [rootdir]
```

**For single files:**

```bash
 python3 ./wav2png.py single --filename [filename.wav]
```

Scaling is done on the STFT output to be compatible with 8-bit png format. The script searches the dataset for the maximum and minimum values, rounds up and down to the nearest integer respectively then scales to [0, 255].

To convert individual png spectrograms back to wav. Script assumes (and inverts) similar scaling as in wav2png. Griffin-Lim algortihm is initialized with [SPSI](http://ieeexplore.ieee.org/abstract/document/7251907/).

**For single png spectrogram:**

```bash
python3 ./png2wav.py [filename.png]
```

## cqtconv

This script is to transform a linear frequency scaled spectrogram image (like the ones generated by wav2png) to a pseudo constant-Q (CQT) scaled spectrogram. Algorithm is adapted from [Matlab code written by Dan Ellis](http://www.ee.columbia.edu/ln/rosa/matlab/sgram/).

**For linear to cqt:**

```bash
python ./cqtconv.py --rootdir [rootdir] --outdir [outdir] --conversion spec2cqt
```

**For cqt to linear:**

```bash
python ./cqtconv.py --rootdir [rootdir] --outdir [outdir] --conversion cqt2spec
```
