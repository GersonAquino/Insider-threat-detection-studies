# TCN–Transformer insider threat detection

Research code for the CERT r4.2 study: a critical reimplementation/adaptation of
Ye et al. (2025), *Daily insider threat detection with hybrid TCN transformer
architecture*. The pipeline prepares daily features and labels, builds
sequences, trains the model and evaluates held-out predictions. Optuna provides
supplementary hyperparameter search; default training retains the selected
manual configuration.

## Setup

Reference environment: Ubuntu 24.04, Python 3.12.3, PyTorch 2.11.0+cu128.
The dependency files pin the reference package versions. From the repository root:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
~~~

CUDA remains the default training device and requires a compatible NVIDIA
driver. Hardware availability is checked when training, HPO or evaluation
executes. Configure DEVICE explicitly in src/itd_tcnTransformer/config/settings.py
for a different runtime; equivalent results on other environments are not established.

## Dataset and paths

Obtain CERT r4.2 and its answers separately from the official CMU/SEI
distribution. No dataset, trained model or experiment output is included.
The loader expects:

~~~text
data/raw/CERT/
  r4.2/
    device.csv
    email.csv
    file.csv
    http.csv
    logon.csv
    psychometric.csv
    LDAP/*.csv
  answers/                 # "Answers" is also supported
    insiders.csv
    r4.2*/...              # scenario answer CSVs
~~~

The loader reads LDAP and psychometric files even though profile features are
disabled in the selected model. Preserve the original dataset notices.

Intermediates default to data/processed/itd_tcnTransformer and outputs to
outputs/itd_tcnTransformer. Set TCN_DATA_ROOT or TCN_OUTPUT_ROOT before Python
starts to relocate the base directories:

~~~bash
export TCN_DATA_ROOT=/path/to/data
export TCN_OUTPUT_ROOT=/path/to/outputs
~~~

Relative overrides resolve from this repository's root, independently of the
working directory. The raw/CERT and processed layout stays below TCN_DATA_ROOT;
itd_tcnTransformer stays below TCN_OUTPUT_ROOT.

## Pipeline

Run these commands from the repository root, in order. They perform real
processing, training and evaluation; they are not installation checks.

~~~bash
python -m src.itd_tcnTransformer.scripts.01_validate_raw_data
python -m src.itd_tcnTransformer.scripts.02_build_daily_features
python -m src.itd_tcnTransformer.scripts.03_build_daily_labels
python -m src.itd_tcnTransformer.scripts.04_build_final_daily_dataset
python -m src.itd_tcnTransformer.scripts.05_build_sequences
python -m src.itd_tcnTransformer.scripts.06_train_model
python -m src.itd_tcnTransformer.scripts.07_evaluate_model
~~~

The first command validates, loads and explores the raw inputs. It accepts
--load-row-chunk-size, --load-keep-old-files, --skip-validation and
--skip-exploration. Experimental settings are in config/settings.py.

The selected model uses two-stage training, four TCN blocks with dilations
1/2/4/8, two Transformer layers and dimension 128. Windows contain 30 consecutive
observed user-day records, which need not be consecutive calendar days.
Profile features and the Transformer attention mask are disabled. Pooled
window_timestep evaluation includes retrospectively contextualised earlier
positions. The evaluator also supports separate final-position metrics.
Thresholds maximise validation recall subject to precision of at least 0.70.

Optional supplementary HPO:

~~~bash
python -m src.itd_tcnTransformer.scripts.run_optuna_search
~~~

HPO stores its SQLite study and trial outputs below the output directory and
resumes a study with the same name. Its default budget is 50 trials.
The training command accepts --params-json /path/to/best_params.json when
overrides are explicitly intended; default training does not adopt HPO output.

## Tests

requirements-dev.txt adds pytest. Included tests cover synthetic evaluation
logic, model shapes, runtime metadata and training startup. Some execute model
forward passes. They do not reproduce the thesis experiments.
