# Polygonal Discontinuous Galerkin (in deal.II)

[![GitHub CI](https://github.com/fdrmrc/Polydeal/actions/workflows/tests.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/tests.yml)
[![Indent](https://github.com/fdrmrc/Polydeal/actions/workflows/indentation.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/indentation.yml)
[![Doxygen](https://github.com/fdrmrc/Polydeal/actions/workflows/doxygen.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/doxygen.yml)


***PolyDEAL*** is an open source library which aims to provide building blocks for the developement of Polygonal Discontinuous Galerkin methods, using the Finite Element library [**deal.II**](https://dealii.org). It is written in C++ using the C++17 standard. 


## Getting started and prerequisites

We require a **cmake** version greater than 2.8.
Furthermore, we have successfully compiled and tested our library with the following compilers:
-  **gcc** versions  >= 9.
-  **clang** versions >= 15
-  **icc** (Intel compiler) 2021.2



The library **polyDEAL** employs **deal.II** as main third-party library. As **deal.II** itself depends on other external libraries for many functionalities, we strongly suggest to download and install deal.II following the instructions available at https://www.dealii.org/download.html and https://www.dealii.org/developer/readme.html. The project as a whole depends on the following list of *mandatory* external libraries which can be configured as dependencies of deal.II during the installation phase:
### METIS
This graph-partitioner *METIS* can be used to partition a triangulation among several processors. In the context of polytopal methods, it has been heavily employed as an agglomeration strategy to build polytopic elements out of a fine grids composed by standard shapes.

### p4est
*p4est* is a library that manages meshes that are distributed across multiple processors using a space-filling curve approach. 

### Trilinos
*Trilinos* (in particular its multilevel solvers and distributed matrices) is employed as main parallel linear algebra library.


To enable to computation of some quality metrics, mostly of theoretical interests and not really relevant in application codes, the external library **CGAL** is required. As this is a dependency of *deal.II* as well, it is sufficient to configure deal.II with it. (`DEAL_II_WITH_CGAL=ON`).

## Building polyDEAL 
Assuming deal.II (version 9.5 onwards) is installed on your machine and meets the requirements above, all is required to do is:

```bash
git clone git@github.com:fdrmrc/Polydeal.git
cd Polydeal/
mkdir build
cd build/
cmake -DDEAL_II_DIR=/path/to/deal.II ..
make -j<N>
```
being N is the number of jobs you want to use to compile. 

## Documentation
A Doxygen generated documentation is built and deployed at each merge to the main branch. You can find the latest documentation here: [https://fdrmrc.github.io/Polydeal/](https://fdrmrc.github.io/Polydeal/).

## Authors and Contact

This project is developed and maintained by:
- [Marco Feder](https://www.math.sissa.it/users/marco-feder) ([@fdrmrc](https://github.com/fdrmrc)), SISSA, IT

under the supervision of 
- [Prof. Andrea Cangiani](https://people.sissa.it/~acangian/) ([@andreacangiani](https://github.com/andreacangiani)), SISSA, IT
- [ Prof. Luca Heltai](https://sites.google.com/view/luca-heltai) ([@luca-heltai](https://github.com/luca-heltai)), University of Pisa, IT


Feel free to start a [discussion](https://github.com/fdrmrc/Polydeal/discussions) or open an [issue](https://github.com/fdrmrc/Polydeal/issues), especially if you want to contribute. For any other inquiries or special requests, you can directly contact Marco Feder (mfeder@sissa.it).