About this project
==================

**PyFlow** is a general-purpose Visual Dataflow Programming library. Nodes represent algorithms with certain inputs and outputs. Connections transfer data from the output (source) of the first node to the input (sink) of the second one.

**PyFlowOpencv** is a visual scripting extension for PyFlow for OpenCV. **PyFlowOpencv** enable you learn Computer vision without writing a single line of code. A quick demo on how **PyFlowOpenCv** works for a face detection.

.. image:: res/quick_demo.gif

Goal
********

Learning OpenCV is quite challenging for most of the beginners. PyFlowOpenCv make the learning curve of OpenCv much smoother. You do not need to write any code, just drag and drop the diagram. 

Installation
==================
PyFlowOpenCv is not a standalone software, it is a extension package of PyFlow. PyFlow has to be installed first. You can refer to `PyFlow <https://github.com/wonderworks-software/PyFlow>`_  to install PyFlow.
After PyFlow installed through pip or setup.py.

The easy way to install PyFlow is::

    pip install git+https://github.com/wonderworks-software/PyFlow.git@master

Clone or download repository to a local folder::

    git clone https://github.com/bobosky/PyFlowOpenCv

Install requirements for your use case::

    pip install -r requirements.txt

To run the program in standalone mode, run pyflow.py in the root folder of PyFlow project. You can also invoke pyflow.exe on windows or pyflow on unix OS. Program is located inside PATH/TO/PYTHON/Scripts.

You can enable the PyFlowOpenCv package by this way.

* Copy the PyFlowOpenCv package to .PyFlow/Packages
* User can add location of package to env variable for PYTHONPATH
* Paths listed in PYFLOW_PACKAGES_PATHS env variable (; separated string)
* addition package on preferences dialog.


.. image:: res/add_pyflowopencv_path.png

If everything works out, you should able to see 'PyFlowOpenCv' in your NodeBox dialog of the GUI.

.. image:: res/all_window.png



Getting Started
==========================

We have `documentation <https://pyflow.readthedocs.io/en/latest/>`_


Authors
=========

**Pedro Cabrera** - `Pedro Cabrera <https://github.com/pedroCabrera>`_ 

**Changbo Yang** - `Changbo Yang <https://github.com/bobosky>`_

See also the list of `contributors <https://github.com/wonderworks-software/PyFlow/contributors>`_ who participated in this project.

Discussion
==============

Join us to our `discord channel <https://discord.gg/SwmkqMj>`_ and ask anything related to project!


.. Nodes
.. ==========

.. Pins
.. ==========

.. Open an image
.. =====================

.. Open a video file 
.. =====================

.. Open a webcam 
.. =====================

.. Basic image processing
.. =========================

.. Image filter
.. ===============

.. Color Conversion
.. ===================

.. Keypoint detection and feature extraction
.. ===============

.. Deep learning Modules
.. ===============