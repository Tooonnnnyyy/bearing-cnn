# Python environment

Python is the interpreter that executes the script. NumPy holds numeric arrays; PyTorch provides tensors, neural-network layers, automatic differentiation and optimizers; Matplotlib produces figures. `csv`, `json`, `argparse`, `pathlib`, `random` and `copy` belong to Python's standard library and need no separate pip installation.

`pip` installs distributions into an interpreter's environment. A virtual environment isolates project packages; it is not a separate computer, container or GPU. `requirements.txt` pins three direct dependencies to the reference versions. It is not a complete cross-platform lockfile: Python, transitive dependencies, OS and hardware can still differ.

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

If `ModuleNotFoundError` appears, first verify the interpreter. Installing into one environment while VS Code executes another is a common cause. Network/DNS errors are installation connectivity failures, not proof that the Python package is unavailable. Select the environment explicitly in VS Code; a shell prompt displaying `(base)` does not override a fully qualified Python executable.

## Local CPU and Colab

The reference CPU run already works. Colab provides a remote Python runtime, sometimes with GPU access subject to availability and usage limits. A VS Code extension connects an editor to a runtime; it does not transfer packages, files or local paths automatically.

Optional Colab workflow, not executed as part of this repository's validation:

1. Create a private notebook and choose a GPU runtime when useful.
2. Clone this public code repository into the runtime. Review existing package versions before replacing Colab's preinstalled PyTorch.
3. Make the nine recordings available through a private upload or private Drive location only if you have permission to send them to that cloud service. Keep sharing private; do not put them in the public Git repository.
4. Install the needed packages into the notebook kernel. `%pip install -r bearing-cnn/requirements.txt` may replace preinstalled packages; a restart may be needed. Prefer an isolated environment for strict reference-version work.
5. Verify `torch.cuda.is_available()`, then run `!python bearing-cnn/bearing_fault_cnn.py --data-dir /content/data --output-dir /content/run --device cuda`.
6. Save allowed results before the runtime is reclaimed. Record the actual environment and hardware.

CUDA deterministic operations can require a workspace setting such as `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python; follow the exact installed PyTorch/CUDA diagnostic. GPU and CPU need not produce bit-identical results. The included code's CPU path is verified; Colab, CUDA and MPS paths are not verified here.

Training speed and model quality are different questions. More GPU compute does not itself repair mislabeled data, recording-level differences or a weak evaluation design.

References: [venv](https://docs.python.org/3/library/venv.html), [pip](https://pip.pypa.io/en/stable/user_guide/), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [Colab FAQ](https://research.google.com/colaboratory/faq.html).
