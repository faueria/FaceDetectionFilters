cmake_minimum_required(VERSION 3.8)

project(VideoDisplay)

# OpenCV path
set(OpenCV_DIR "C:/Users/Nihal Sandadi/Downloads/opencv/build")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "C:/Users/Nihal Sandadi/Downloads/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib/lib")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

add_executable(VideoDisplay 
    vidDisplay.cpp 
    filters.cpp
    faceDetect.cpp
)

target_link_libraries(VideoDisplay ${OpenCV_LIBS} onnxruntime)

# for the Haar cascade file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_frontalface_alt2.xml
    ${CMAKE_CURRENT_BINARY_DIR}/haarcascade_frontalface_alt2.xml
    COPYONLY
)

# making sure everything works
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")