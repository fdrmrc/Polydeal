cmake_minimum_required(VERSION 3.13.4)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAGGLO_DEBUG")

FIND_PACKAGE(deal.II QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
ENDIF()

# Check the found version and ensure it meets our requirements
MESSAGE(STATUS "Found deal.II version: ${deal.II_VERSION}")
MESSAGE(STATUS "deal.II installation path: ${deal.II_DIR}")
IF(${deal.II_VERSION} VERSION_LESS "9.7.0")
  MESSAGE(FATAL_ERROR "\n"
    "*** deal.II version ${deal.II_VERSION} found, but version 9.7.0 or newer is required. ***\n\n"
    "Please update your deal.II installation to version 9.7.0 or newer."
  )
ELSE()
  MESSAGE(STATUS "deal.II version ${deal.II_VERSION} meets the minimum requirement (>= 9.7.0)")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(polydeal)

# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(test/polydeal)
