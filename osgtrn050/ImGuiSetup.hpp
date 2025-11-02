#pragma once
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include <GL/glew.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <osg/OperationThread>   // <-- correct header for osg::Operation
#include <osg/GraphicsContext>
#include <iostream>

// =============== Initialize ImGui & GLEW safely =================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}

    void operator()(osg::Object *object) override
    {
        auto *gc = dynamic_cast<osg::GraphicsContext *>(object);
        if (!gc)
            return;

        if (glewInit() != GLEW_OK)
        {
            std::cerr << "[ImGuiInitOperation] GLEW initialization failed!\n";
            return;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        if (!ImGui_ImplOpenGL3_Init("#version 130"))
            std::cerr << "[ImGuiInitOperation] ImGui_ImplOpenGL3_Init() failed!\n";
        else
            std::cout << "[ImGuiInitOperation] ImGui initialized successfully.\n";
    }
};

// =============== Shutdown operation (optional cleanup) ===============
class ImGuiShutdownOperation : public osg::Operation
{
public:
    ImGuiShutdownOperation() : osg::Operation("ImGuiShutdownOperation", false) {}

    void operator()(osg::Object *) override
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext();
        std::cout << "[ImGuiShutdownOperation] ImGui shut down.\n";
    }
};