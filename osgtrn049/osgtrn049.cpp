// ==== IMPORTANT: choose GLEW as ImGui's OpenGL loader ====
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include <GL/glew.h>
// =========================================================

#include <osg/Camera>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/OrbitManipulator>
#include <osgViewer/Viewer>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"   // your handler that calls ImGui::NewFrame()/Render() each frame
#include "CommonFunctions"       // osgCookBook::createAnimationPathCallback(...)


// ======================= ImGui Init ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        auto *gc = dynamic_cast<osg::GraphicsContext *>(object);
        if (!gc)
            return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

static osg::Vec3d worldCenterOf(osg::Node* node)
{
    if (!node) return osg::Vec3d();
    const auto& paths = node->getParentalNodePaths();
    if (paths.empty()) return osg::Vec3d();
    osg::Matrix l2w = osg::computeLocalToWorld(paths.front());
    return node->getBound().center() * l2w;
}

class FollowOrbitManipulator : public osgGA::OrbitManipulator
{
public:
    explicit FollowOrbitManipulator(osg::Node* target)
        : _target(target), _offset(0.0, -80.0, 25.0), _alignYaw(true) {}

    void setOffset(const osg::Vec3d& off) { _offset = off; }
    void setAlignYaw(bool enable) { _alignYaw = enable; }

    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override
    {
        bool handled = osgGA::OrbitManipulator::handle(ea, aa);

        if (_target.valid() && ea.getEventType() == osgGA::GUIEventAdapter::FRAME)
        {
            const auto& paths = _target->getParentalNodePaths();
            if (!paths.empty())
            {
                osg::Matrix world = osg::computeLocalToWorld(paths.front());
                osg::Vec3d targetPos = world.getTrans();
                osg::Quat rotation = world.getRotate();

                osg::Vec3d center = targetPos;
                osg::Vec3d eye = _alignYaw ? targetPos + rotation * _offset
                                           : targetPos + _offset;

                setCenter(center);
                setHomePosition(eye, center, osg::Vec3d(0, 0, 1));
            }
        }
        return handled;
    }

private:
    osg::observer_ptr<osg::Node> _target;
    osg::Vec3d _offset;
    bool _alignYaw;
};

class ManipulatorControlPanel : public OsgImGuiHandler
{
public:
    ManipulatorControlPanel(osgGA::KeySwitchMatrixManipulator* ks,
                            FollowOrbitManipulator* follow)
        : _keySwitch(ks), _follow(follow), _selected(0),
          _dist(80.0f), _height(25.0f), _alignYaw(true) {}

    void drawUi() override
    {
        ImGui::Begin("Camera Manipulator");
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
            changed |= ImGui::SliderFloat("Height",   &_height, 5.0f,  80.0f);
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
    int   _selected;
    float _dist, _height;
    bool  _alignYaw;
};

int main(int argc, char** argv)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::MatrixTransform> trans = new osg::MatrixTransform;
    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");
    trans->addUpdateCallback(osgCookBook::createAnimationPathCallback(100.0f, 20.0f));
    trans->addChild(model.get());

    osg::ref_ptr<osg::MatrixTransform> terrain = new osg::MatrixTransform;
    terrain->addChild(osgDB::readNodeFile(dataPath + "lz.osg"));
    terrain->setMatrix(osg::Matrix::translate(0.0, 0.0, -200.0));

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(terrain.get());
    root->addChild(trans.get());

    osg::ref_ptr<osgGA::OrbitManipulator> orbit = new osgGA::OrbitManipulator;

    osg::ref_ptr<osgGA::NodeTrackerManipulator> nodeTracker = new osgGA::NodeTrackerManipulator;
    nodeTracker->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);
    nodeTracker->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);
    nodeTracker->setTrackNode(trans.get());

    osg::ref_ptr<FollowOrbitManipulator> follow = new FollowOrbitManipulator(trans.get());

    osg::Vec3d center0(0,0,0);
    osg::Vec3d eye0(0.0, -60.0, 25.0);
    osg::Vec3d up0(0,0,1);
    orbit->setHomePosition(eye0, center0, up0);
    nodeTracker->setHomePosition(eye0, center0, up0);
    follow->setHomePosition(eye0, center0, up0);

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keySwitch = new osgGA::KeySwitchMatrixManipulator;
    keySwitch->addMatrixManipulator('1', "Orbit",       orbit.get());
    keySwitch->addMatrixManipulator('2', "NodeTracker", nodeTracker.get());
    keySwitch->addMatrixManipulator('3', "FollowOrbit", follow.get());
    keySwitch->selectMatrixManipulator(0);

    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 900, 700);
    viewer.setSceneData(root.get());
    viewer.setCameraManipulator(keySwitch.get());

    viewer.setRealizeOperation(new ImGuiInitOperation);

    // ImGui panel as an event handler (your OsgImGuiHandler is a GUIEventHandler subclass)
    osg::ref_ptr<ManipulatorControlPanel> imguiPanel =
        new ManipulatorControlPanel(keySwitch.get(), follow.get());

    viewer.addEventHandler(imguiPanel.get());

    return viewer.run();
}