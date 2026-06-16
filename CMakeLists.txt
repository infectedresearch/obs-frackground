cmake_minimum_required(VERSION 3.28)

project(obs-frackground VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBOBS REQUIRED libobs)
find_library(ONNXRUNTIME_LIBRARY REQUIRED NAMES onnxruntime)

add_library(obs-frackground MODULE
  src/inference_worker.cpp
  src/model_adapter.cpp
  src/onnx_backend.cpp
  src/plugin.cpp
)

target_compile_options(obs-frackground PRIVATE -Wall -Wextra)
target_include_directories(obs-frackground PRIVATE
  /usr/include/obs
  /usr/include/onnxruntime
  ${LIBOBS_INCLUDE_DIRS}
)
target_link_directories(obs-frackground PRIVATE ${LIBOBS_LIBRARY_DIRS})
target_link_libraries(obs-frackground PRIVATE ${LIBOBS_LIBRARIES} ${ONNXRUNTIME_LIBRARY})

set_target_properties(obs-frackground PROPERTIES
  PREFIX ""
  OUTPUT_NAME "obs-frackground"
)

install(TARGETS obs-frackground
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins"
)

install(DIRECTORY data/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-frackground"
)

install(FILES models/rvm_mobilenetv3_fp16.onnx
  OPTIONAL
  DESTINATION "${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-frackground"
)

add_executable(frackground-probe
  tools/probe_model.cpp
  src/model_adapter.cpp
  src/onnx_backend.cpp
)

target_compile_options(frackground-probe PRIVATE -Wall -Wextra)
target_include_directories(frackground-probe PRIVATE
  /usr/include/onnxruntime
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(frackground-probe PRIVATE ${ONNXRUNTIME_LIBRARY})
