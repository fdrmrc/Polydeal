# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(polydeal SHARED agglomeration_handler.cc)
DEAL_II_SETUP_TARGET(polydeal)

