# external project base directory for all projects in this hierarchy
set_property(DIRECTORY PROPERTY EP_BASE ${DEPENDENCIES_DIR})

# Download TorchLib if no path was specified
add_subdirectory(torch)

add_subdirectory(viennaps)
