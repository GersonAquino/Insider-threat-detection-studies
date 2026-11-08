# SPEDIA LSTM study

Source for the preliminary `balanced50_test` SPEDIA experiment: event preparation,
window generation, LSTM training, Keras Tuner HPO, calibration and evaluation.
Datasets, trained models and historical results are not included.

## Linux usage

Use Python 3.12 and run these commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SPEDIA_CSV_PATH="/path/to/logs_SPEDIA_annotated_en_extended_6m_x6_balanced.csv"
export SPEDIA_OUTPUT_ROOT="./outputs/my_run"
python -m itd.pipeline.spedia_prepare_job
python -m itd.pipeline.spedia_train_eval_job
```

The second command runs HPO, final training and evaluation; it can be expensive.
Use a fresh output root for each execution. Preparation refuses an occupied
fixed-window directory; training retains the original HPO/checkpoint behaviour.
TensorFlow GPU/CUDA setup depends on the execution environment.

Both environment variables accept absolute paths or paths relative to this study
directory. Defaults are `data/raw/SPEDIA/logs_SPEDIA_annotated_en_extended_6m_x6_balanced.csv`
and `outputs`. Other settings are in `src/itd/config.py`.

## Retained protocol

The default experiment uses 90-minute user-windows, the last 150 events per
window, and a positive window label when any retained event is positive. Temporal
cuts use 70%/15%/15% of the observed time range. The model combines five hashed
categorical embeddings, 12 event numerical features, one LSTM and 25 window
features. Training normalisers are fitted on training data.

Keras Tuner random search maximises validation AUPRC over 30 trials with two
executions per trial. The top three candidates are compared using seeds 1337,
2027 and 4242. The final model uses work-hours sample weighting, validation-fitted
Platt calibration and a validation threshold targeting precision 0.80. Segmented
thresholds are supplementary; hard-negative mining and dynamic-window HPO are
disabled for the retained experiment. Optional source paths remain available
through the configuration.

## Reproducibility limits

The required input is a local extended/balanced SPEDIA derivative, not an official
upstream dataset release. Its complete transformation and augmentation lineage
is unavailable, so these sources do not reconstruct that derivative from the
original SPEDIA data. Required CSV columns are `User`, `Timestamp` and `Anomaly`;
the adapter handles additional event, domain, attachment and size fields.

Python 3.12.3 and Keras 3.13.0 appear in retained metadata, but the exact historical
TensorFlow version and full dependency lock were not preserved. Requirements
therefore describe execution dependencies, not an exact historical environment.
The exported code has been checked statically; no training or evaluation was run
as part of this source export.
