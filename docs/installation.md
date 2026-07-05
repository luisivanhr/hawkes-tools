# Installation

## From GitHub

Install the current `master` branch directly from GitHub:

```powershell
python -m pip install "hawkes-tools @ git+https://github.com/luisivanhr/hawkes-tools.git"
```

Verify the import package:

```powershell
python -c "import hawkes_tools; print(hawkes_tools.__file__)"
```

## From A Local Checkout

From the repository root:

```powershell
python -m pip install -e .
```

For development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Optional PyTorch-backed cumulant functionality uses the `torch` extra:

```powershell
python -m pip install -e ".[torch]"
```

## Requirements

The package currently requires Python 3.11 or newer and these runtime
dependencies:

- `numpy>=2.0`
- `scipy>=1.13`
- `matplotlib>=3.8`
- `numba>=0.60`

Numba is required. If it is missing, the package is not considered correctly
installed.

## Package Identity

Use this import:

```python
import hawkes_tools
```

Do not import through the previous migration namespace. The library is
standalone and does not install a compatibility namespace for the previous
compiled package.

## Editable Development Environment

On the maintainer Windows workstation, the validated interpreter has been:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

For public users, use the active Python environment where `hawkes-tools` is
installed.
