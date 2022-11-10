if(DEFINED Torch_DIR AND NOT DEFINED CACHE{Torch_DIR})
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${Torch_DIR}" Torch_DIR)
  set(Torch_DIR
      "${Torch_DIR}"
      CACHE PATH "Path to libtorch installation" FORCE)
  # Now look for the dependency at the specified path
  find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)
  message(STATUS "Found LibTorch at ${Torch_DIR}.")
  add_library(torch_external INTERFACE) # dummy target
else()
  # First try to locate any local build
  find_package(Torch QUIET PATHS "${DEPENDENCIES_DIR}/Install/torch_external" NO_DEFAULT_PATH)

  # If that fails look for a system installation
  if(NOT Torch_FOUND)
    find_package(Torch QUIET)
  endif()
  if(Torch_FOUND)
    message(STATUS "Found LibTorch at ${Torch_DIR}.")
    add_library(torch_external INTERFACE) # dummy target
  else()
    message(
      STATUS
        "Couldn't find LibTorch. Dependency will be built with the buildDependencies target (e.g. `make buildDependencies`)."
    )

    ExternalProject_Add(
      torch_external
      URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE 1
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1)

    # Get install and build directory from external project
    ExternalProject_Get_Property(torch_external SOURCE_DIR)

    set(Torch_DIR "${SOURCE_DIR}")

    # add to buildDependencies target
    add_dependencies(buildDependencies torch_external)
  endif()
  set(Torch_DIR
      "${Torch_DIR}"
      CACHE PATH "Path to libtorch installation" FORCE)
  set(Torch_INCLUDEDIR
      "${Torch_DIR}/include"
      CACHE PATH "Path to libtorch include directories" FORCE)
  set(Torch_LIBRARYDIR
      "${Torch_DIR}/lib"
      CACHE PATH "Path to libtorch library directories" FORCE)
endif()
