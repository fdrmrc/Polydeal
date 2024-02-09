# Polygonal Discontinuous Galerkin (in deal.II)

[![GitHub CI](https://github.com/fdrmrc/Polydeal/actions/workflows/tests.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/tests.yml)
[![Indent](https://github.com/fdrmrc/Polydeal/actions/workflows/indentation.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/indentation.yml)
[![Doxygen](https://github.com/fdrmrc/Polydeal/actions/workflows/doxygen.yml/badge.svg)](https://github.com/fdrmrc/Polydeal/actions/workflows/doxygen.yml)


***PolyDEAL*** is an open source project written in C++ which aims to provide building blocks for the implementation of Polygonal Discontinuous Galerkin methods, within the Finite Element library [**deal.II**](https://dealii.org).


## Getting started
Assuming deal.II (version 9.5 onwards) is installed on your machine and you have a compiler C++17-compliant compiler, all is required to do is:
```bash
git clone git@github.com:fdrmrc/Polydeal.git
cd Polydeal/
mkdir build
cd build/
cmake -DDEAL_II_DIR=/path/to/deal.II ..
make -j<N>
```


## Documentation
A Doxygen generated documentation is built and deployed at each merge to the main branch. You can find the latest documentation here: [https://fdrmrc.github.io/Polydeal/](https://fdrmrc.github.io/Polydeal/).

## Authors

This project is maintained by:
- [Marco Feder](https://www.math.sissa.it/users/marco-feder) ([@fdrmrc](https://github.com/fdrmrc)), SISSA, IT
- [Luca Heltai](http://people.sissa.it/~heltai) ([@luca-heltai](https://github.com/luca-heltai)), SISSA, IT
- [Andrea Cangiani](https://people.sissa.it/~acangian/) ([@andreacangiani](https://github.com/andreacangiani)), SISSA, IT
