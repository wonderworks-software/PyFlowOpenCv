# PyFlowOpenCv
**PyFlowOpencv** is a visual scripting extension for PyFlow. 


Learning OpenCV is quite challenging for most of the beginners. PyFlowOpenCv make the learning curve of Opencv much smoother. You do not need to write any code, just drag and drop the diagram. 


# Installation
PyFlowOpenCv is not a standalone software, it is a extension package of PyFlow. PyFlow has to be installed first. You can refer to `PyFlow <https://github.com/wonderworks-software/PyFlow>`_  to install PyFlow.
After PyFlow installed through pip or setup.py.

Clone or download repository
```bash
    git clone https://github.com/wonderworks-software/PyFlowOpenCv
```
Install requirements for your use case::

```bash
    pip install -r requirements.txt
```
To run the program in standalone mode, run pyflow.py in the root folder of PyFlow project. You can also invoke pyflow.exe on windows or pyflow on unix OS. Program is located inside PATH/TO/PYTHON/Scripts.

You can enable the PyFlowOpenCv package by one of the following ways.

- Copy the PyFlowOpenCv package to .PyFlow/Packages
- User can add location of package to env variable for PYTHONPATH
- Paths listed in PYFLOW_PACKAGES_PATHS env variable (; separated string)
addition package on preferences dialog.
- To enable the PyFlowOpenCv package, user can add location of package to env variable for PYTHONPATH, or via pyflow editor itself in preferences.

 ![addpackage](docs/source/res/add_pyflowopencv_path.png)

If everything works out, you should able to see 'PyFlowOpenCv' in your NodeBox dialog of the GUI.

 ![gui](docs/source/res/all_window.png)

