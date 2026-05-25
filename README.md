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

## Files

- `notebooks/cc_bos_gpt4o_reproduction_colab.ipynb`: runnable Colab notebook for smoke and full GPT-4o reproduction runs.
- `examples/ccbos_smoke.csv`: safe 5-row CSV for checking that the Colab/API pipeline runs.
- `scripts/create_ccbos_reproduction_notebook.py`: generator used to create the notebook.
