cmake_minimum_required(VERSION 3.5)
project(osgtrn036)

# ---- Find packages ----
find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgGA osgViewer osgText)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# ---- Common ImGui path ----
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/../_deps/imgui-src)

include_directories(
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# ---- ImGui sources ----
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp      # optional
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ---- Project sources ----
set(SOURCES
    osgtrn036.cpp
    OsgImGuiHandler.cpp
    ${IMGUI_SOURCES}
)

add_executable(${PROJECT_NAME} ${SOURCES})

# ---- Link ----
target_link_libraries(${PROJECT_NAME}
    ${OPENSCENEGRAPH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
)