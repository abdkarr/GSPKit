# GSPKit

`GSPKit` is a python package that includes some functions that I have been using
when working on research projects on graph data. Instead of keep copying these
functions into different projects, I decided to create a repo that could be
simply installed to the environment. You are welcomed to utilize it in your
projects if any of these functionalities help you. 

The package is by no means a complete package that can be used for graph
analysis or graph signal processing. I might extend the package with time, but
for more complete toolboxes, you can check [networkx](https://networkx.org/),
[GSPBox](https://epfl-lts2.github.io/gspbox-html/) etc.

## Installation 

Use `pip` to install the package from Github: 
```bash
$ pip install git+https://github.com/abdkarr/GSPKit.git
```

Minimum supported python version is `3.10`.

### Development

The development of this packege is done by `poetry`, which needs to be installed 
if one wants to work with the development version of the packege. To install 
poetry please see this [link](https://python-poetry.org/docs/) (installation 
with `pipx` is preferred). Once `poetry` is installed, clone the repo in an
appropriate directory and `cd` into it:
```bash
$ git clone https://github.com/SPLab-aviyente/fastSGL.git
$ cd fastSGL
```
Next, create a virtual environment. If you have `conda` installed, this can 
be done as follows:
```bash
$ conda create -n GSPKit -c conda-forge python
```
Activate the newly created environment and install the package:
```bash
$ conda activate GSPKit
(GSPKit) $ poetry install --all-extras
```
`poetry` installs the editable version of the package to your environment. Thus,
any change you make to the package is reflected to your environment. 