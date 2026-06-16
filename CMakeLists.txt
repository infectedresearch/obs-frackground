cmake_minimum_required(VERSION 3.28)

project(obs-frackround VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBOBS REQUIRED libobs)
find_library(ONNXRUNTIME_LIBRARY REQUIRED NAMES onnxruntime)

add_library(obs-frackround MODULE
  src/inference_worker.cpp
  src/model_adapter.cpp
  src/onnx_backend.cpp
  src/plugin.cpp
)

target_compile_options(obs-frackround PRIVATE -Wall -Wextra)
target_include_directories(obs-frackround PRIVATE
  /usr/include/obs
  /usr/include/onnxruntime
  ${LIBOBS_INCLUDE_DIRS}
)
target_link_directories(obs-frackround PRIVATE ${LIBOBS_LIBRARY_DIRS})
target_link_libraries(obs-frackround PRIVATE ${LIBOBS_LIBRARIES} ${ONNXRUNTIME_LIBRARY})

set_target_properties(obs-frackround PROPERTIES
  PREFIX ""
  OUTPUT_NAME "obs-frackround"
)

install(TARGETS obs-frackround
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins"
)

install(DIRECTORY data/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-frackround"
)

install(FILES models/rvm_mobilenetv3_fp16.onnx
  OPTIONAL
  DESTINATION "${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-frackround"
)

add_executable(frackround-probe
  tools/probe_model.cpp
  src/model_adapter.cpp
  src/onnx_backend.cpp
)

target_compile_options(frackround-probe PRIVATE -Wall -Wextra)
target_include_directories(frackround-probe PRIVATE
  /usr/include/onnxruntime
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(frackround-probe PRIVATE ${ONNXRUNTIME_LIBRARY})
