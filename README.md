# Decompile LLM
An LLM powered Python decompiler, that restores source code from `.pyc` files.

## Introduction
This program aims to restore human-readable source code from `.pyc` files. decompile-llm uses the power of AI to restore source code for all Python versions, and all future Python versions.

## Why
As Python evolves, decompilation has become more difficult, as other tools are no longer supporting newer Python versions. Fortunately, LLMs have an intimate understanding of Python bytecode, allowing this tool to automatically reconstruct source codes from every python version. However, it is important to note that, since we are utilizing LLMs, the accuracy of the reconstruction varies depending on your target code, and the model you choose.

I recommend using [decompyle6](https://github.com/rocky/python-uncompyle6) or [decompile3](https://github.com/rocky/python-decompile3) if the reconstruction is not working as expected, or you would like a more traditional way to decompile.

Since the accuracy heavily depends on the model you choose, I noted down below the current most capable model. 

---

## Requirements

- `.pyc` file
- Python 3.10 or higher
- OpenAI API key or Google Gemini API key

---

## Features

- Works with `.pyc` files from **all** Python versions
- Automatically disassembles bytecode using **xdis**
- Supports multiple LLM providers
  - **OpenAI** (GPT-4.1 by default)
  - **Google Gemini** (Gemini 2.5 Flash by default; free tier available)
- Syntax verification of decompiled code
- Smart chunking for large files
- Progress bar
- Streamed outputs

---

## Installation

The most straightforward way to install is via pypi.
```bash
pip install decompilellm
```

You can also install from source by cloning the repository and installing dependencies manually.
```bash
# Clone the repository
git clone https://github.com/iancdev/python-decompile-llm
cd python-decompile-llm

# Install required dependencies
pip install -r requirements.txt
```
---

## Quick Start

The code is designed to run out of the box, without much configuration. Advanced arguments are provided, but you can get started with the below basic commands.

Python version is automatically detected for disassembly. 

### Most Capable and Cost Optimized (Free Tier Available)
Use this for the best code output for free.
This option uses Gemini 2.5 Flash with reasoning effort set to high, and runs 3 iterations.
```bash
decompilellm --provider google --key <gemini_key> --verify yes --iter 3 --output decompiled.py --effort high <targetfile>.pyc
```
> Free tier Gemini model requests may be used by Google to train their models. **Always** review the provider's terms of service before use.
### Using GPT-4.1 (Default OpenAI Model)

```bash
decompilellm --key <openai_key> <targetfile>.pyc
```

### Using Gemini 2.5 Flash (Free Tier Available, default Google Model)

```bash
decompilellm --provider google --key <gemini_key> <targetfile>.pyc
```

### Save output to a file

```bash
decompilellm --key <openai_key> --output decompiled.py (targetfile>.pyc
```
---

## Advanced options
For advanced users seeking additional control over the program's behavior, you may apply the following advanced options.
> You may also run `decompilellm --help` to view all available options.

| Flag | Purpose | Default |
|------|---------|---------|
| `--model MODEL` | Which LLM model to use (e.g. `gpt-4.1`, `gemini-2.5-pro`, etc.). | Provider-specific |
| `--key KEY` | API key for the chosen provider (overrides env vars). | – |
| `--systemmsg MSG` | Custom system prompt for the decompiler LLM. | Built-in prompt |
| `--iter N` | Number of times to run the model and keep the best answer. | `1` |
| `--verify {yes,no}` | Check Python syntax of the decompiled code. | `yes` |
| `--retry N` | How many extra attempts to make if a run fails verification. | `0` |
| `--output FILE` | Write result to `FILE` instead of stdout. | stdout |
| `--stream / --no-stream` | Stream tokens live to the terminal. Enabled by default unless writing to a file. | on |
| `--multithreaded / --no-multithreaded` | Run iterations in parallel threads. | on |
| `--threads N` | Explicit thread count when multithreading. | Same as `--iter` (capped) |
| `--provider {chatgpt,gpt,gemini,google,openai}` | Backend to hit. | `openai` |
| `--split N` | Manually break the disassembly into **N** equal-sized chunks. Overrides `--auto-split`. | `0` (off) |
| `--auto-split` | Automatically chunk large byte-code based on `--max-tokens`. | off |
| `--max-tokens N` | Target tokens per chunk when auto-splitting. Requires `tiktoken`. | `10000` |
| `--max-chars N` | Fallback char length per chunk if token counts aren’t available. | `50000` |
| `--temp FLOAT` | Sampling temperature (0.0–2.0 OpenAI, 0.0–1.0 Gemini). | `0.5` |
| `--topp FLOAT` | Nucleus-sampling `top_p` value. | `1.0` |
| `--effort {none,low,medium,high}` | Hint for reasoning depth; higher can improve accuracy at a cost. | `none` |

---

## Extracting `.pyc` from PyInstaller executables
For PyInstaller generated binaries, you can use **pyinstxtractor** to retrieve the embedded `.pyc` file.
You can use Detect-It-Easy or similar tools to detect which compilation method was used.

* [pyinstxtractor (classic, Python)](https://github.com/extremecoders-re/pyinstxtractor)
* [pyinstxtractor-ng (actively maintained, Python 3)](https://github.com/pyinstxtractor/pyinstxtractor-ng)
* [pyinstxtractor-go](https://github.com/pyinstxtractor/pyinstxtractor-go) **or** [pyinstxtractor-web](https://pyinstxtractor-web.netlify.app/)

---

## Environment variables

Instead of passing API keys on the command line you can set:

* `OPENAI_API_KEY` - For OpenAI models
* `GEMINI_API_KEY` - For Google Gemini models

This is recommended for repeat usage and is overall more secure.

---
## FAQ + Troubleshooting

###  The output code does not reproduce the same functionality as the original code.

This is expected in most cases, since some information is lost during compilation. The reconstruction may include optimizations or other changes that are not in the original source code, causing some functionality to break. You are encouraged to use other tools to decompile as well for verification and getting a better understanding of the original code.

### The output code is incorrect.

As mentioned above, the output code may be incorrect in cases where the information is simply lost. However, for major errors, you can try to increase accuracy by choosing a reasoning model, and setting reasoning effort to high. Additionally, you can get better output by increasing the iteration amount to >=3, and using syntax verification.

The below is a copy and paste command for Gemini-2.5 Pro with reasoning effort on high, running 5 parallel sessions.

```bash
decompilellm --provider google --key <gemini_key> --verify yes --iter 5 --effort high --multithreaded --model gemini-2.5-pro <targetfile>.pyc
```

### The output code is incomplete.

This can happen when the automated splitting does not detect the correct model for token calculation, or no split occured even though it is supposed to. You can correct this by either using a model with higher context winddow support, or by manually specifying splits. Below is a command line for manual splitting into 5 chunks. 
*Please note that chunks are split evenly, which may cause artifacts in the output code. This may cause syntax verification to fail. You may choose to disable verification and manually fix the final output for these cases.*

```bash
decompilellm --provider google --key <gemini_key> --verify yes --iter 3 --multithreaded --split-manual 5 <targetfile>.pyc
```

> For long projects, it's currently recommended to use Gemini models with higher context windows, and to split manually (since tiktoken does not support automated token calculation for Google Gemini models).

### The output code is empty.

Check if your `pyc` file is valid. If your target file Python version is > 3.13, you may need to manually update xdis to the latest version.

### I have another issue

You can open an issue on GitHub, and I can try to help. Please make sure to include as much detail as possible within your issue.

---

## Notes

* The quality of decompilation depends on both the complexity of the code, the model you're using, and the reasoning effort you chose (if any).
* For especially complex code, consider GPT-o3 or Gemini 2.5 Pro or Flash models with reasoning effort set to high. (This can get costly!)
* Alternatives: [uncompyle6](https://github.com/rocky/python-uncompyle6) and [decompile3](https://github.com/rocky/python-decompile3)

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for details.
