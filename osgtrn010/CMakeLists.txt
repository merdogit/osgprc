cmake_minimum_required(VERSION 3.5)

project(osgtrn010)

# Set executable name
set(EXECUTABLE_NAME osgtrn010)

# Find OpenSceneGraph
find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgGA osgViewer osgText)

# Find OpenGL, GLEW, GLFW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# ImGui include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/backends
)

# ImGui sources
set(IMGUI_SOURCES
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/imgui.cpp
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/imgui_widgets.cpp
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/imgui_demo.cpp      # optional
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/backends/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/_deps/imgui-src/backends/imgui_impl_opengl3.cpp
)

# Project sources
set(SOURCES
    osgtrn010.cpp
    OsgImGuiHandler.cpp
    ${IMGUI_SOURCES}
)

# Create executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${EXECUTABLE_NAME}
    ${OPENSCENEGRAPH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
)