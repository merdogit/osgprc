#pragma once
#include <osgGA/KeySwitchMatrixManipulator>
#include "FollowOrbitManipulator.hpp"
#include "OsgImGuiHandler.hpp"
#include <imgui.h>

//
// ManipulatorControlPanel
// -----------------------
// ImGui panel for switching between manipulators and
// controlling FollowOrbit parameters.
//
class ManipulatorControlPanel : public OsgImGuiHandler
{
public:
    ManipulatorControlPanel(osgGA::KeySwitchMatrixManipulator* ks,
                            FollowOrbitManipulator* follow)
        : _keySwitch(ks), _follow(follow),
          _selected(0), _dist(80.0f), _height(25.0f), _alignYaw(true)
    {
    }

    void drawUi() override
    {
        ImGui::Begin("Camera Controller");
        ImGui::Text("Select Camera Mode:");
        const char* modes[] = {"Orbit", "NodeTracker", "FollowOrbit"};

        for (int i = 0; i < 3; ++i)
        {
            if (ImGui::RadioButton(modes[i], _selected == i))
            {
                _selected = i;
                _keySwitch->selectMatrixManipulator(i);
            }
        }

        if (_selected == 2)
        {
            ImGui::Separator();
            ImGui::Text("FollowOrbit Settings");
            bool changed = false;
            changed |= ImGui::SliderFloat("Distance", &_dist, 20.0f, 200.0f);
            changed |= ImGui::SliderFloat("Height", &_height, 5.0f, 80.0f);

            if (changed)
                _follow->setOffset(osg::Vec3d(0.0, -_dist, _height));

            if (ImGui::Checkbox("Align with Yaw", &_alignYaw))
                _follow->setAlignYaw(_alignYaw);
        }

        ImGui::End();
    }

private:
    osg::observer_ptr<osgGA::KeySwitchMatrixManipulator> _keySwitch;
    osg::observer_ptr<FollowOrbitManipulator> _follow;
    int _selected;
    float _dist, _height;
    bool _alignYaw;
};
