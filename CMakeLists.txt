cmake_minimum_required(VERSION 3.5)
project(calibrate_camera)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibrate_camera main.cpp)
target_link_libraries(calibrate_camera ${OpenCV_LIBS})

file(COPY camera_cal DESTINATION ./)
