# Build both Release and Debug versions of library and executables
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

file(GLOB APP_SOURCES ./*.cc )

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")
      
      # Pick up the right library
      set(_lib "polydeal${_p}")

      foreach (testsourcefile ${APP_SOURCES})
      # Cut off the file extension and directory path
            get_filename_component( exename ${testsourcefile} NAME_WE )            
            # Add suffix for debug
            set(testname ${exename}${_p})
            message("-- Configuring executable ${testname} (${testsourcefile})")

            add_executable( ${testname} ${testsourcefile} )

            # Make sure YourLib is linked to each app
            target_link_libraries(${testname} ${_lib})
            DEAL_II_SETUP_TARGET(${testname} ${_BUILD_TYPE})
            INSTALL(TARGETS ${testname})
            set(testname ${testname}${_p})
            endforeach( testsourcefile ${APP_SOURCES} )
        endif ()
endforeach()
