cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ViennaPS-LibTorch-Example)

set(CMAKE_CXX_STANDARD "17")

# set default build type
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Check dependencies
include(ExternalProject)
set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
set(STAMPFILE ${CMAKE_BINARY_DIR}/target.stamp)
# When the buildDependencies target is executed, a stamp file is touched
add_custom_target(buildDependencies COMMAND ${CMAKE_COMMAND} -E touch ${STAMPFILE})
add_subdirectory(external/upstream)

# Create the initial stamp file
file(TOUCH ${STAMPFILE})
# Include the stamp file, so that cmake is forced to re-run once the file has been touched
include(${STAMPFILE})

# Try to find all dependencies. If all are found, include the main project's cmake file, otherwise
# display a message.
find_package(Torch QUIET PATHS ${Torch_DIR} NO_DEFAULT_PATH)
find_package(ViennaPS QUIET PATHS ${ViennaPS_DIR} NO_DEFAULT_PATH)
if(Torch_FOUND AND ViennaPS_FOUND)
  # Remove the buildDependencies target from the ALL target to prevent unnecessary re-builds
  set_target_properties(buildDependencies PROPERTIES EXCLUDE_FROM_ALL true)
  add_subdirectory(src)
else()
  message(WARNING "Not all dependencies were found. Execute buildDependencies target first.")
  # Add the buildDependencies target to the ALL target
  set_target_properties(buildDependencies PROPERTIES EXCLUDE_FROM_ALL false)
endif()
