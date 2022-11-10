if(DEFINED ViennaPS_DIR AND NOT DEFINED CACHE{ViennaPS_DIR})
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${ViennaPS_DIR}" ViennaPS_DIR)
  set(ViennaPS_DIR
      "${ViennaPS_DIR}"
      CACHE PATH "Path to ViennaPS installation" FORCE)
  # Now look for the dependency at the specified path
  find_package(ViennaPS REQUIRED PATHS ${ViennaPS_DIR} NO_DEFAULT_PATH)
  message(STATUS "Found ViennaPS at ${ViennaPS_DIR}")
  add_library(viennaps_external INTERFACE) # dummy target
else()
  # First try to locate any local build
  find_package(ViennaPS QUIET PATHS "${DEPENDENCIES_DIR}/Install/viennaps_external" NO_DEFAULT_PATH)

  # If that fails look for a system installation
  if(NOT ViennaPS_FOUND)
    find_package(ViennaPS QUIET)
  endif()

  if(ViennaPS_FOUND)
    message(STATUS "Found ViennaPS at ${ViennaPS_DIR}.")
    add_library(viennaps_external INTERFACE) # dummy target
  else()
    message(
      STATUS
        "Couldn't find ViennaPS. Dependency will be built with the buildDependencies target (e.g. `make buildDependencies`)."
    )
    ExternalProject_Add(
      viennaps_external
      GIT_REPOSITORY https://github.com/yozoon/ViennaPS.git
      GIT_TAG geometric
      GIT_SHALLOW 1
      INSTALL_DIR "${DEPENDENCIES_DIR}/Install/viennaps_external"
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                 -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
                 -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --target buildDependencies all
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1
      GIT_PROGRESS 1
      EXCLUDE_FROM_ALL TRUE)

    # Get install and build directory
    ExternalProject_Get_Property(viennaps_external INSTALL_DIR)

    set(ViennaPS_DIR "${INSTALL_DIR}")

    # add to buildDependencies target
    add_dependencies(buildDependencies viennaps_external)
  endif()
  set(ViennaPS_DIR
      ${ViennaPS_DIR}
      CACHE PATH "Path to ViennaPS installation" FORCE)
endif()
