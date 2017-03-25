project(stereovis)
cmake_minimum_required(VERSION 3.1.0)

# add CMakeLists.txt source dir and cmake module dir to the cmake module path
# cmake modules are cmake files used e.g. to find library files
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/CMakeModules"
)

# specify runtime dir to be the build dir for out-of-source build
# (aka shadow build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

### EXTERNAL LIBRARIES ###

find_package(OpenCV REQUIRED)

### COMPILER FLAGS ###

# gcc flags
set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

### SOURCE FILES ###

# relative path to every single h and cpp file
set(SRC_CLASSES
    src/main.cpp
)
		
# adds an executable target with given name 
# to be built from the source files listed afterwards
add_executable(${PROJECT_NAME} ${SRC_CLASSES})

### INCLUDE HEADER FILES ###
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

### LINK LIBRARIES ###
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

### COPY DATA ###

add_custom_target(data)

add_custom_command(TARGET data PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/images
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/images
)

add_dependencies(${PROJECT_NAME} data)
