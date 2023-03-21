# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(Agglomeration_suite SHARED agglomeration_handler.cc non_nested_transfer.cc)
DEAL_II_SETUP_TARGET(Agglomeration_suite)
