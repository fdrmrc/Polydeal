# # tests/support/CMakeLists.txt

# SET(TEST_LIBRARIES polydeal)
# DEAL_II_PICKUP_TESTS()

# set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")

  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
    DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
    MESSAGE("-- Found ${_build_type} version of deal.II.")

    # Setup library
    SET(_lib "polydeal${_p}")
    MESSAGE("-- Configuring test for ${_lib}")

    SET(TEST_LIBRARIES ${_lib})
    DEAL_II_PICKUP_TESTS()
    set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endforeach()
