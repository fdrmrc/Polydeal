# adapted from https://vicrucann.github.io/tutorials/quick-cmake-doxygen/

# check if Doxygen is installed
find_package(Doxygen)

if(DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # copy input to output file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("-- Configured to build Doxygen documentation")

  # create documentation
  add_custom_target(doxygen
    COMMAND sed -i "s|INPUT = include|INPUT = ${CMAKE_CURRENT_SOURCE_DIR}/../../include|g" ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings.out 2>&1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Doxygen documentation"
    VERBATIM)
endif(DOXYGEN_FOUND)
