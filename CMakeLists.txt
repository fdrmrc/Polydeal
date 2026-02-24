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

# Trilinos, METIS, and p4est are required
IF(NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_METIS OR NOT DEAL_II_WITH_P4EST)
  IF(NOT DEAL_II_WITH_TRILINOS)
    MESSAGE(STATUS "  - Trilinos: NOT found (required)")
  ENDIF()
  IF(NOT DEAL_II_WITH_METIS)
    MESSAGE(STATUS "  - METIS: NOT found (required)")
  ENDIF()
  IF(NOT DEAL_II_WITH_P4EST)
    MESSAGE(STATUS "  - p4est: NOT found (required)")
  ENDIF()
  MESSAGE(FATAL_ERROR "\n"
    "*** deal.II is missing required dependencies (see above). ***\n\n"
    "Polydeal requires deal.II to be compiled with Trilinos, METIS, and p4est.\n"
    "Please recompile deal.II with -DDEAL_II_WITH_TRILINOS=ON -DDEAL_II_WITH_METIS=ON -DDEAL_II_WITH_P4EST=ON."
  )
ELSE()
  MESSAGE(STATUS "Trilinos, METIS, and p4est support detected.")
ENDIF()

# Define mesh directory path
SET(MESH_DIR "${CMAKE_SOURCE_DIR}/meshes")
MESSAGE(STATUS "Mesh directory: ${MESH_DIR}")

# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(test/polydeal)
