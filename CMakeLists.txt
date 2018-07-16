cmake_minimum_required(VERSION 3.5)
project(calibrate_camera)

set(CMAKE_CXX_STANDARD 11)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(OpenCV_DIR C:/Users/gw889026/CLionProjects/opencv)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibrate_camera main.cpp)
target_link_libraries(calibrate_camera stdc++fs ${OpenCV_LIBS})

file(COPY camera_cal DESTINATION ./)