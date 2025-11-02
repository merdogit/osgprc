cmake_minimum_required(VERSION 3.5)

# Project name
project(osgtrn008)

# Set the executable name as a variable
set(EXECUTABLE_NAME osgtrn008)

# Find OpenSceneGraph components
find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgGA osgViewer)

# Source files (you can add more later)
set(SOURCES
    osgtrn008.cpp
)

# Create executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${EXECUTABLE_NAME}
    ${OPENSCENEGRAPH_LIBRARIES}
)