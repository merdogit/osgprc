#pragma once
#include "/home/murate/Documents/SwTrn/DearIamGuiTrn/imgui/imgui.h"
#include "/home/murate/Documents/SwTrn/DearIamGuiTrn/imgui/backends/imgui_impl_opengl3.h"
#include <osgViewer/Viewer>

class ImGuiOSG
{
public:
    static void init() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplOpenGL3_Init("#version 130");
    }

    static void newFrame() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui::NewFrame();
    }

    static void render() {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    static void shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext();
    }
};
