cmake_minimum_required(VERSION 3.8)
project(pedestrian_detection_c4)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(OpenCV_DIR /Users/wengguifan/opencv-2.4.13/release)
include_directories("/usr/local/Cellar/opencv@2/2.4.13.2_2/include" "include" )
FIND_PACKAGE(OpenCV REQUIRED)

#set(SOURCE_FILES main.cpp )
set(SOURCE_FILES main.cpp include/C4Feature.hpp include/IntImage.hpp include/C4Trainer.hpp include/C4Detector.hpp)
add_executable(pedestrian_detection_c4 ${SOURCE_FILES})


target_link_libraries(pedestrian_detection_c4 ${OpenCV_LIBS})