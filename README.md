# LinguaJailbreak-Lab

Swarm-guided discovery and analysis of cross-lingual jailbreak attacks in large language models.

## Current Notebook

[Open the CC-BOS Colab notebook](https://colab.research.google.com/github/batis1/LinguaJailbreak-Lab/blob/main/notebooks/cc_bos_gpt4o_reproduction_colab.ipynb)

The default mode is now `qwen_only`, which uses Qwen-Plus for prompt generation, target response, translation, and judging. Use it first to verify the full CC-BOS pipeline with one working API provider.

The notebook also keeps `strict_gpt4o_reproduction` for the public CC-BOS implementation's native setting:

- Method: CC-BOS
- Language: Classical Chinese
- Target model: GPT-4o
- Prompt-generation model: DeepSeek-Chat
- Translation model: DeepSeek-Chat
- Judge model: GPT-4o
- Population size: 5
- Maximum iterations: 5
- Final success criterion: released-code score >= 80
- Released-code early stop: score >= 120

The notebook does not include the AdvBench CSV. Upload a `goal,intention` CSV in Colab. If the uploaded file is not exactly the paper's 50-query subset, the run metadata labels it as a reconstructed or smoke subset.

## Colab Secrets

Create these Google Colab Secrets before running:

- `QWEN_API_KEY` for the default Qwen-only run
- `QWEN_BASE_URL` (optional; defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `OPENAI_API_KEY` and `DEEPSEEK_API_KEY` only for `strict_gpt4o_reproduction`
- `OPENAI_BASE_URL` (optional; leave unset for the default OpenAI endpoint)
- `DEEPSEEK_BASE_URL` (optional; defaults to `https://api.deepseek.com`)

## AgentTrapBench Colab Outputs

The AgentTrapBench MVP notebook mounts Google Drive when `SAVE_TO_GOOGLE_DRIVE = True` and writes self-contained run artifacts to:

- `MyDrive/AgentTrapBench/runs/<run_id>/`
- `MyDrive/AgentTrapBench/latest/`

Each run folder includes the synthetic dataset, rendered web/mobile assets, raw results CSV, summary CSV, plot, and metadata JSON.

## AgentTrapBench Deep v2

[Open the Deep v2 Colab notebook](https://colab.research.google.com/github/batis1/LinguaJailbreak-Lab/blob/main/notebooks/mobile_web_agenttrapbench_deep_v2_colab.ipynb)

Deep v2 expands the MVP into a benchmark scaffold with open-dataset ingestion, source-family coverage reports, and a discrete swarm optimizer for mock-only indirect prompt-injection attacks. Start with `RUN_PROFILE = "qwen_swarm_smoke"` to verify the Qwen-VL path, then use `RUN_PROFILE = "qwen_deep"` for a serious run.

When `SAVE_TO_GOOGLE_DRIVE = True`, Deep v2 writes versioned artifacts to:

- `MyDrive/AgentTrapBenchDeepV2/runs/<run_id>/`
- `MyDrive/AgentTrapBenchDeepV2/latest/`

Important artifacts include `open_dataset_load_report.csv`, `base_dataset.csv`, `swarm_history.csv`, `best_genome.json`, `optimized_attack_dataset.csv`, `results_<run_id>.csv`, `summary_<run_id>.csv`, `source_breakdown_<run_id>.csv`, and `target_report_<run_id>.json`.

## Files

- `notebooks/cc_bos_gpt4o_reproduction_colab.ipynb`: runnable Colab notebook for smoke and full GPT-4o reproduction runs.
- `notebooks/mobile_web_agenttrapbench_mvp_colab.ipynb`: Colab-ready MVP scaffold for the Mobile-Web AgentTrapBench AI-agent security benchmark.
- `notebooks/mobile_web_agenttrapbench_deep_v2_colab.ipynb`: deeper Colab benchmark with open dataset loaders and swarm-guided mock attack search.
- `examples/ccbos_smoke.csv`: safe 5-row CSV for checking that the Colab/API pipeline runs.
- `scripts/create_ccbos_reproduction_notebook.py`: generator used to create the notebook.
- `scripts/create_agenttrapbench_mvp_notebook.py`: generator used to create the AgentTrapBench MVP notebook.
- `scripts/create_agenttrapbench_deep_v2_notebook.py`: generator used to create the AgentTrapBench Deep v2 notebook.
- `research_briefs/agent_security_ideas_2026-06-04.md`: ranked research brief for mobile/web AI-agent attack ideas and the recommended first project.
