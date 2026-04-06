# Preprocessing Cheatsheet

This is the shortest preprocessing explanation you need for this repo.

## One-Sentence Version

"For each EEG file, the code keeps EEG channels, applies average reference, estimates the spectrum with Welch PSD from 1 to 30 Hz, and turns each channel into bandpower features such as alpha or beta power."

## What Average Reference Means

Average reference means:

- take the average signal across all kept EEG channels at each time point
- subtract that average from each channel

Plain-English explanation:

"Instead of measuring each channel against one specific reference electrode, the signal is re-expressed relative to the average of all channels."

In this repo, that happens in `load_and_preprocess_eeg()` inside `src/train_lemon_multitarget.py`.

## What Welch PSD Means

PSD means power spectral density.

Plain-English version:

"It estimates how much signal power is present at different frequencies."

Welch PSD is a stable way to do that:

- split the signal into shorter chunks
- compute a spectrum for each chunk
- average those spectra

That reduces how noisy a single spectrum estimate would be.

## Is FFT Involved?

Yes.

Welch PSD is built from FFT-based spectral estimates under the hood. You do not need to explain the FFT math in detail for this project.

A good interview answer is:

"Yes, Welch PSD is an FFT-based way to estimate spectral power more stably by averaging across windows."

## What Bandpower Means

Bandpower means:

- choose a frequency band such as alpha
- add up the PSD values inside that frequency range

In this repo, bands include:

- delta: 1-4 Hz
- theta: 4-8 Hz
- alpha: 8-12 Hz
- beta: 12-30 Hz

So bandpower is just:

"How much total power does this channel have inside this named frequency range?"

## How One Feature Is Created

Worked example: `f3_alpha_eo`

1. Start with the subject's EO file, such as `sub-032_EO.set`
2. Read it with MNE
3. Keep EEG channels only
4. Apply average reference
5. Take the F3 channel
6. Estimate Welch PSD from 1 to 30 Hz
7. Look at the alpha band, 8-12 Hz
8. Integrate the PSD over that band with `np.trapz`
9. Save that number as `f3_alpha_eo`

Feature-name meaning:

- `f3` = channel
- `alpha` = band
- `eo` = eyes-open condition

## What the Code Actually Uses

In `src/train_lemon_multitarget.py`:

- `load_and_preprocess_eeg()` handles EEG-only channel selection and average reference
- `compute_bandpower_features()` calls `mne.time_frequency.psd_array_welch(...)`
- `np.trapz(...)` integrates the PSD values inside each frequency band

## Why This Is Enough for an Interview Explanation

You do not need to explain advanced DSP theory for this repo.

This is enough:

"I used a simple spectral feature pipeline. For each EO and EC recording, I applied average reference, estimated the spectrum with Welch PSD, integrated power inside standard EEG bands, and used those bandpower values as tabular features for benchmarking."

That answer is technically correct, short, and directly tied to the code in this repository.
