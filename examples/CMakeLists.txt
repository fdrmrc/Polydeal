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
            
            # Require MUMPS for monodomain examples
            if(exename STREQUAL "monodomain_DG3D" OR exename STREQUAL "monodomain_DG2D")
              if(NOT DEAL_II_WITH_MUMPS)
                message(WARNING "\n"
                  "*** The example '${exename}' requires MUMPS support. ***\n\n"
                  "Please recompile deal.II with MUMPS enabled for this example to work correctly.\n"
                  "To enable MUMPS in deal.II, use -DDEAL_II_WITH_MUMPS=ON when configuring deal.II."
                )
              else()
                message("-- MUMPS support detected for ${exename}")
              endif()
            endif()
            
            # Add suffix for debug
            set(testname ${exename}${_p})
            message("-- Configuring executable ${testname} (${testsourcefile})")

            add_executable( ${testname} ${testsourcefile} )

            # Make sure the library is linked to each app
            target_link_libraries(${testname} ${_lib})
            
            # Define MESH_DIR for the executable
            target_compile_definitions(${testname} PRIVATE MESH_DIR="${MESH_DIR}")
            
            DEAL_II_SETUP_TARGET(${testname} ${_BUILD_TYPE})
            INSTALL(TARGETS ${testname})
            set(testname ${testname}${_p})
            endforeach( testsourcefile ${APP_SOURCES} )
        endif ()
endforeach()
