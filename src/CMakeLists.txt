# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(Agglomeration_suite SHARED)
target_sources(${PROJECT_NAME}
  PRIVATE
  agglomeration_handler.cc
  agglomerator.cc
)
DEAL_II_SETUP_TARGET(Agglomeration_suite)