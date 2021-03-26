cmake_minimum_required(VERSION 3.17)
project(stereo_slam)

set(OpenCV_DIR /Users/linyicheng/Documents/quad/opencv/install/lib/cmake/opencv4)

set(CMAKE_CXX_STANDARD 17)

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package(OpenCV REQUIRED)
find_package(PCL 1.5 REQUIRED)

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable(stereo_slam main.cpp)

# tell it to link the executable target against OpenCV
target_link_libraries(stereo_slam ${OpenCV_LIBS} ${PCL_LIBRARIES})