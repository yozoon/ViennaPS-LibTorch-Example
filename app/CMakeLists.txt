set(Demo "TorchDemo")

add_executable(${Demo} ${Demo}.cpp)

target_include_directories(${Demo} PUBLIC ${VIENNAPS_INCLUDE_DIRS})
target_link_libraries(${Demo} ${TORCH_LIBRARIES})
target_link_libraries(${Demo} ${VIENNAPS_LIBRARIES})

configure_file(scripts/pretrained_model.pt ${CMAKE_CURRENT_BINARY_DIR}/pretrained_model.pt COPYONLY)
configure_file(scripts/train.py ${CMAKE_CURRENT_BINARY_DIR}/train.py COPYONLY)
configure_file(scripts/scatterdata.csv ${CMAKE_CURRENT_BINARY_DIR}/scatterdata.csv COPYONLY)
