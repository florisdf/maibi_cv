[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/florisdf/maibi_cv/HEAD)

# Computer Vision - MAIBI

This repository contains the notebooks for the course of Computer Vision, given 
at the master AI for Business and Industry.

# Getting started on your own pc

## Install Python 3.6 and pip

If this is not yes installed, see <https://www.python.org/downloads/> for
instructions.

Also, make sure pip is installed. See
<https://pip.pypa.io/en/stable/installation/> for details.

## Set up the virtual environment

Open a terminal window and clone this repository

```bash
git clone https://github.com/florisdf/maibi_cv.git
```

Change the directory of your terminal to move into the repository.

```bash
cd maibi_cv
```

Create a new virtual environment.

```bash
python3.6 -m venv .venv
```

**Note**: it is possible that this command fails to run because `venv` is not installed yet. For Ubuntu, this can be installed with

```bash
sudo apt install python3.6-venv
```

Activate the environment you just created:

```bash
source .venv/bin/activate
```

Also, make sure pip has the most recent version.

```bash
pip install --upgrade pip
```

Finally, install the requirements into the virtual environment.

```bash
pip install -r requirements.txt
```

## Open Jupyter Notebook

Run the following command in your terminal. **Make sure that**:

1. You are in the directory of this repository (so in the directory called
   `maibi_cv`).
2. You have **activated the virtual environment** with `source .venv/bin/activate`.

```bash
jupyter notebook
```
