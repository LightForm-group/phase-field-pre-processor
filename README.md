Python code to generate CIPHER input files from a MatFlow DAMASK simulation.

This code is a **work in progress**

# Installation

- You will need to install the Matlab python package by following the instructions [here](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

- Note that you must run this code with a version of Python 3 that is compatible with your version of Matlab, as shown [here](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf).

- The best way to use this code is by setting up a virtual environment and installing all the packages listed in `requirements.txt`, plus the Matlab Python package. You can set up a virtual environment like this (make sure `python` is a version of python that is compatible with your version of Matlab):

```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Then follow the instructions linked above to install the Matlab Python package; make sure you do it within your new virtual environment.

Alternatively, it is also possible to use a conda environment.

# Notebook

After the envrionment is set up and activated, you can launch `jupyter` to run the notebook.
