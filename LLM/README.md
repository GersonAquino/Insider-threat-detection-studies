# Local LLM experiment

Windows pipeline for constructing CERT r4.2 user-day subsets and classifying them through a local Ollama server. Python 3.12.3, DuckDB 1.5.2, PyYAML 6.0.3 and Ollama 0.22.0 were recorded for the completed experiment.

Run from this directory in PowerShell:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3:14b
python scripts/cert_llm_subset/build_subset.py --config scripts/cert_llm_subset/config.qwen.yaml
python scripts/cert_llm_subset/ollama_eval.py --config scripts/cert_llm_subset/config.qwen.yaml --variant day_only
```

Ollama must be running at `http://127.0.0.1:11434`. The completed thesis experiment used `qwen3:14b` (Q4_K_M), five examples and `day_only`. `config.gpt.yaml` preserves the separate preliminary GPT-OSS attempt; `with_context` is implemented but was not evaluated in the completed experiment. Model weights are obtained separately; a model tag alone does not guarantee identical historical weights.

No datasets, subsets, model weights or results are included. Supply these CERT inputs independently:

```text
data/raw/CERT/r4.2/{logon,device,file,email,http,psychometric}.csv
data/raw/CERT/r4.2/LDAP/*.csv
data/raw/CERT/answers/r4.2-*/*.csv
```

Paths in the YAML configuration resolve against this project's root, independently of the current working directory. Pass the configuration filename relative to the working directory, or use an absolute filename. Configure external input paths using `dataset.raw_dir` and `dataset.answers_glob`; `dataset.output_dir` controls generated subsets and evaluation outputs. `CERT_LLM_PROJECT_ROOT` overrides the root. By default, generated files go to `data/llm_subsets/cert_r4.2/user_day_v1`.

Use the explicit Qwen configuration above to retain the thesis parameters: built-in defaults differ in maximum subset size. Existing predictions resume only when their configuration hashes match. `--overwrite` explicitly replaces predictions; use a separate output directory to preserve earlier runs.
