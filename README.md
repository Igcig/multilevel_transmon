# Multilevel transmon

Simple package for diagonalizing a transmon or a transmon–resonator system and for calculating transition matrix elements for:

- the qubit transition between the ground and excited state with n photons in the resonator
- the qubit transition between the ground and second excited state
- the qubit transition between the first and second excited states

## Requirements

- [Rust toolchain](https://rustup.rs)
- Python 3.x+
- `maturin`: `pip install maturin`

## Project Structure

```
multilevel_transmon/
├── transmon_demo.ipynb
├── pyproject.toml
├── README
├── LICENSE
├── transmon_tools/
│   ├── __init__.py
│   └── transmon_tools.py
└── spectra_calculation/
    ├── src/
    │   └── lib.rs
    └── Cargo.toml
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/igcig/multilevel_transmon.git
cd tu-repo
```

### 2. Install the Python package and dependencies
```bash
pip install -e .
```

### 3. Build the Rust extension

From the project root folder:

```bash
cd spectra_calculation
```

**With a virtual environment (recommended):**
```bash
maturin develop
```

**Without a virtual environment:**
```bash
maturin build
pip install ./target/wheels/<wheel_name>.whl
```

## Development

To rebuild after changes to the Rust code, re-run `maturin develop` from the `spectra_calculation/` folder or force reinstall with
```bash
maturin build
pip install --force-reinstall ./target/wheels/<wheel_name>.whl
```