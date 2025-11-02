cmake_minimum_required(VERSION 3.5)

project(osgtrn048)

# Set executable name
set(EXECUTABLE_NAME osgtrn048)

# Find OpenSceneGraph
find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgGA osgViewer osgText)

# Source files
set(SOURCES
    osgtrn048.cpp
    CommonFunctions.cpp
)

# Create executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${EXECUTABLE_NAME} ${OPENSCENEGRAPH_LIBRARIES})
