## Synopsis

KSPTrack is a method for the segmentation of video and volumetric sequences
with sparse point supervision.

![alt text](pipeline.png "pipeline")

## Installation

This software depends on the following independent components that you will have
to install first.
Both make use of the [C++ boost library](https://www.boost.org).
The installation procedure are given in the respective github repositories
as well as here for convenience.

We also provide a docker image at =krakapwa/boost= as well as in
the form of Dockerfile.

### SLIC Supervoxels
simple and efficient supervoxels.

```
git clone https://github.com/lejeunel/SLICsupervoxels
cd SLICsupervoxels
mkdir build
cd build
cmake ..
make
python3 src/setup.py install
```

### Edge-disjoint K-shortest paths
C++ implementation. Uses the boost graph library.

```
git clone https://github.com/lejeunel/boost_ksp
cd boost_ksp
mkdir build
cd build
cmake ..
make
python3 src/setup.py install
```

### Install the whole thing
Once both external dependencies are installed, procede to the current package:
```
git clone https://github.com/lejeunel/KSPTrack
cd KSPTrack
pip install .
pip install -r requirements.txt
```

## Usage
All parameters used in this program are set in `cfgs/cfg.py`.
Among others, it sets the paths to the input frames and 2D locations.

For quick start, see `single_ksp.py`.
