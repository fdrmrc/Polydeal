# set up executable with autopilot macro:
SET(TARGET "poisson")
SET(TARGET_SRC minimal_SIP.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} Agglomeration_suite)
