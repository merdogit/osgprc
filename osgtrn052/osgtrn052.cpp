#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include <GL/glew.h>

#include <osg/Camera>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/OrbitManipulator>
#include <osgGA/NodeTrackerManipulator>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= ImGui Init Operation ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        if (auto *gc = dynamic_cast<osg::GraphicsContext *>(object))
            ImGui_ImplOpenGL3_Init();
    }
};

// ======================= Custom Follow Orbit Manipulator ===========================
class FollowOrbitManipulator : public osgGA::OrbitManipulator
{
public:
    explicit FollowOrbitManipulator(osg::Node *target)
        : _target(target), _offset(0.0, -80.0, 25.0), _alignYaw(true) {}

    void setOffset(const osg::Vec3d &off)
    {
        _offset = off;
        updateCameraPosition();
    }

    void updateCameraPosition()
    {
        if (!_target)
            return;

        const auto &paths = _target->getParentalNodePaths();
        if (paths.empty())
            return;

        osg::Matrix world = osg::computeLocalToWorld(paths.front());
        osg::Vec3d targetPos = world.getTrans();
        osg::Quat rotation = world.getRotate();

        osg::Vec3d eye = _alignYaw ? targetPos + rotation * _offset
                                   : targetPos + _offset;

        setCenter(targetPos);
        setTransformation(eye, targetPos, osg::Vec3d(0, 0, 1));
    }

    void setAlignYaw(bool enable) { _alignYaw = enable; }

    bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa) override
    {
        bool handled = osgGA::OrbitManipulator::handle(ea, aa);

        if (_target.valid() && ea.getEventType() == osgGA::GUIEventAdapter::FRAME)
        {
            const auto &paths = _target->getParentalNodePaths();
            if (!paths.empty())
            {
                osg::Matrix world = osg::computeLocalToWorld(paths.front());
                osg::Vec3d targetPos = world.getTrans();
                osg::Quat rotation = world.getRotate();

                osg::Vec3d eye = _alignYaw ? targetPos + rotation * _offset : targetPos + _offset;
                setCenter(targetPos);
                setHomePosition(eye, targetPos, osg::Vec3d(0, 0, 1));
            }
        }
        return handled;
    }

private:
    osg::observer_ptr<osg::Node> _target;
    osg::Vec3d _offset;
    bool _alignYaw;
};

// ======================= Scene Builder ===========================
osg::ref_ptr<osg::Group> createScene(osg::ref_ptr<osg::MatrixTransform> &planeXform)
{
    osg::ref_ptr<osg::Group> root = new osg::Group;

    // Adjust the path for your system
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Node> terrain = osgDB::readNodeFile(dataPath + "lz.osg");
    osg::ref_ptr<osg::Node> planeModel = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");

    planeXform = new osg::MatrixTransform;
    planeXform->addChild(planeModel.get());
    planeXform->setMatrix(osg::Matrix::translate(0.0, 0.0, 20.0)); // Start above terrain

    root->addChild(terrain.get());
    root->addChild(planeXform.get());

    return root;
}

// ======================= ImGui Camera Panel ===========================
class CameraControlPanel : public OsgImGuiHandler
{
public:
    CameraControlPanel(osgGA::KeySwitchMatrixManipulator *ks,
                       osgGA::OrbitManipulator *orbit,
                       osgGA::NodeTrackerManipulator *tracker,
                       FollowOrbitManipulator *follow)
        : _keySwitch(ks), _orbit(orbit), _tracker(tracker), _follow(follow),
          _selected(0), _distance(80.0f), _height(25.0f), _alignYaw(true) {}

    void drawUi() override
    {
        ImGui::Begin("Camera Control Panel");
        const char *modes[] = {"Orbit", "NodeTracker", "FollowOrbit"};
        for (int i = 0; i < 3; ++i)
        {
            if (ImGui::RadioButton(modes[i], _selected == i))
            {
                syncBeforeSwitch(i);
                _keySwitch->selectMatrixManipulator(i);
                _selected = i;
            }
        }

        if (_selected == 0)
        {
            ImGui::SliderFloat("Orbit Distance", &_distance, 20.0f, 200.0f);
            _orbit->setDistance(_distance);
        }
        if (_selected == 2)
        {
            ImGui::Separator();
            ImGui::Text("=== Follow Orbit Settings ===");
            bool changed = false;
            changed |= ImGui::SliderFloat("Distance", &_distance, 20.0f, 200.0f);
            changed |= ImGui::SliderFloat("Height", &_height, 5.0f, 80.0f);
            if (changed)
                _follow->setOffset(osg::Vec3d(0.0, -_distance, _height));

            if (ImGui::Checkbox("Align with Yaw", &_alignYaw))
                _follow->setAlignYaw(_alignYaw);
        }

        // ==== FIXED CAMERA POSITION BLOCK ====
        osg::Vec3d eye, center, up;
        osgGA::CameraManipulator *active = _keySwitch->getCurrentMatrixManipulator();
        if (auto *orbit = dynamic_cast<osgGA::OrbitManipulator *>(active))
            orbit->getTransformation(eye, center, up);
        else if (auto *tracker = dynamic_cast<osgGA::NodeTrackerManipulator *>(active))
            tracker->getTransformation(eye, center, up);
        else if (auto *follow = dynamic_cast<FollowOrbitManipulator *>(active))
            follow->getTransformation(eye, center, up);

        ImGui::Separator();
        ImGui::Text("=== Camera Position ===");
        ImGui::Text("Eye: (%.2f, %.2f, %.2f)", eye.x(), eye.y(), eye.z());
        ImGui::Text("Center: (%.2f, %.2f, %.2f)", center.x(), center.y(), center.z());
        ImGui::End();
    }

private:
    void syncBeforeSwitch(int nextMode)
    {
        osg::Vec3d eye, center, up;
        osgGA::CameraManipulator *active = _keySwitch->getCurrentMatrixManipulator();
        if (auto *orbit = dynamic_cast<osgGA::OrbitManipulator *>(active))
            orbit->getTransformation(eye, center, up);
        else if (auto *tracker = dynamic_cast<osgGA::NodeTrackerManipulator *>(active))
            tracker->getTransformation(eye, center, up);
        else if (auto *follow = dynamic_cast<FollowOrbitManipulator *>(active))
            follow->getTransformation(eye, center, up);

        if (nextMode == 0)
            _orbit->setTransformation(eye, center, up);
        if (nextMode == 1)
            _tracker->setTransformation(eye, center, up);
        if (nextMode == 2)
            _follow->setHomePosition(eye, center, up);
    }

    osg::observer_ptr<osgGA::KeySwitchMatrixManipulator> _keySwitch;
    osg::observer_ptr<osgGA::OrbitManipulator> _orbit;
    osg::observer_ptr<osgGA::NodeTrackerManipulator> _tracker;
    osg::observer_ptr<FollowOrbitManipulator> _follow;
    int _selected;
    float _distance, _height;
    bool _alignYaw;
};

// ======================= MAIN ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::MatrixTransform> planeXform;
    osg::ref_ptr<osg::Group> root = createScene(planeXform);

    // --- Camera manipulators ---
    osg::ref_ptr<osgGA::OrbitManipulator> orbit = new osgGA::OrbitManipulator;
    osg::ref_ptr<osgGA::NodeTrackerManipulator> tracker = new osgGA::NodeTrackerManipulator;
    osg::ref_ptr<FollowOrbitManipulator> follow = new FollowOrbitManipulator(planeXform.get());

    tracker->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);
    tracker->setTrackNode(planeXform.get());

    osg::Vec3d homeEye(0.0, -80.0, 30.0), homeCenter(0, 0, 0), up(0, 0, 1);
    orbit->setHomePosition(homeEye, homeCenter, up);
    tracker->setHomePosition(homeEye, homeCenter, up);
    follow->setHomePosition(homeEye, homeCenter, up);

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> ks = new osgGA::KeySwitchMatrixManipulator;
    ks->addMatrixManipulator('1', "Orbit", orbit.get());
    ks->addMatrixManipulator('2', "NodeTracker", tracker.get());
    ks->addMatrixManipulator('3', "FollowOrbit", follow.get());
    ks->selectMatrixManipulator(0);

    // --- Viewer setup ---
    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 1000, 700);
    viewer.setSceneData(root.get());
    viewer.setCameraManipulator(ks.get());
    viewer.setRealizeOperation(new ImGuiInitOperation);

    // --- ImGui panel ---
    osg::ref_ptr<CameraControlPanel> panel =
        new CameraControlPanel(ks.get(), orbit.get(), tracker.get(), follow.get());
    viewer.addEventHandler(panel.get());

    return viewer.run();
}