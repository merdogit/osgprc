cmake_minimum_required(VERSION 3.5)

project(osgtrn005)

set(EXECUTABLE_NAME osgtrn005)

# Find OpenSceneGraph
find_package(OpenSceneGraph REQUIRED COMPONENTS osgDB osgUtil osgGA osgViewer osgText)

# Find OpenGL and GLFW
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Source files
set(SOURCES
    osgtrn005.cpp
    # ImGui core
    ../../DearIamGuiTrn/imgui/imgui.cpp
    ../../DearIamGuiTrn/imgui/imgui_draw.cpp
    ../../DearIamGuiTrn/imgui/imgui_widgets.cpp
    ../../DearIamGuiTrn/imgui/imgui_tables.cpp
    # ImGui backends
    ../../DearIamGuiTrn/imgui/backends/imgui_impl_glfw.cpp
    ../../DearIamGuiTrn/imgui/backends/imgui_impl_opengl3.cpp
)

# Create executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Include directories for ImGui
target_include_directories(${EXECUTABLE_NAME} PRIVATE
    ../../DearIamGuiTrn/imgui
    ../../DearIamGuiTrn/imgui/backends
)

# Link libraries
target_link_libraries(${EXECUTABLE_NAME} PRIVATE
    ${OPENSCENEGRAPH_LIBRARIES}
    glfw
    OpenGL::GL
)