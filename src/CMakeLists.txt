# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(Agglomeration_suite SHARED agglomeration_handler.cc)
DEAL_II_SETUP_TARGET(Agglomeration_suite)

# set up executable with autopilot macro:
SET(TARGET "step")
SET(TARGET_SRC step.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} Agglomeration_suite)