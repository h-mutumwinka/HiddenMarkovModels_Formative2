# Human Activity Recognition with a Hidden Markov Model

Formative 2 – Machine Learning Techniques II (ALU)

This project uses a Gaussian Hidden Markov Model, implemented from scratch in NumPy, to infer four activities (**Still, Standing, Walking, Jumping**) from smartphone accelerometer and gyroscope data recorded with the Sensor Logger app.

## Results

| | Accuracy |
|---|---|
| Unseen test session (2 test sequences, 137 windows) | **100 %** |
| Leave-one-session-out (all 5 sessions, 664 windows) | **98.9 %** |

## Repository structure

```
hmm.ipynb   full pipeline, already executed with all outputs
data/
  raw/<activity>/<activity>_NN/  Sensor Logger exports (Accelerometer, TotalAcceleration, Gyroscope, Metadata .csv)
  clean/<activity>/<activity>_NN.csv   harmonized 50 Hz recordings (one timestamp column, all sensors aligned)
  test_sequences/test_sequence_{1,2}.csv   the two unseen test sequences (session 5)
  recordings_manifest.csv        recording -> activity, session, original file name, recording time
figures/                         plots and the metrics table used in the report
```

## Data

* 4 activities × 50 recordings of ~5–9 s each (≈ 5 min per activity), recorded in 5 sessions of 10 recordings
* Individual project: one participant, one phone (Samsung Galaxy S10, SM-G973F, Android), 100 Hz nominal; harmonized to **50 Hz**
* Columns in `data/clean`: `timestamp` (Unix s), `seconds`, `lin_*` (linear acceleration, m/s²), `acc_*` (total acceleration incl. gravity, m/s²), `gyro_*` (rad/s), `activity`

## Method

1. **Harmonization:** align the sensors on a common time span, trim 0.5 s at each end, apply a 20 Hz Butterworth anti-alias filter, and resample to 50 Hz.
2. **Windows:** 2 s (100 samples) with 50 % overlap.
3. **Features (9):** 5 time-domain (acc std, SMA, range, gyro RMS, y–z correlation) and 4 FFT-based (dominant frequency, spectral energy, spectral entropy, gyro dominant-bin power). Intensity features are log-transformed, then all features are Z-scored using statistics from the training set only.
4. **HMM:** 4 states with full-covariance Gaussian emissions. The model is initialized from labelled windows, then refined with Baum–Welch, which stops when the change in log-likelihood falls below ε = 10⁻³. Decoding uses Viterbi.
5. **Evaluation:** session 5 is held out completely. Metrics are sensitivity, specificity, accuracy and a confusion matrix, plus leave-one-session-out cross-validation.

## Running

```bash
pip install numpy pandas scipy matplotlib
jupyter notebook hmm.ipynb   # Run All, about 30 s
```

Tested with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18 and Matplotlib 3.11.
