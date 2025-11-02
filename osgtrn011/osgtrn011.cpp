#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/GUIEventHandler>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// --- Camera view enum ---
enum class CameraView
{
    Chase,
    Front,
    Top
};

// ---------------- PlaneMotion Class ----------------
class PlaneMotion
{
public:
    PlaneMotion(osg::ref_ptr<osg::MatrixTransform> _planeTransform)
        : _planeTransform(_planeTransform),
          _roll(0.0f), _pitch(0.0f), _yaw(0.0f),
          _posX(0.0f), _posY(0.0f), _posZ(0.0f),
          _circleEnabled(false), _circleRadius(20.0f),
          _circleSpeed(1.0f), _angle(0.0f)
    {}

    void update()
    {
        if (_circleEnabled)
        {
            _angle += 0.01f * _circleSpeed;
            _posX = _circleRadius * cos(_angle);
            _posY = _circleRadius * sin(_angle);
        }

        osg::Quat _qRoll(osg::DegreesToRadians(_roll), osg::Vec3(1,0,0));
        osg::Quat _qPitch(osg::DegreesToRadians(_pitch), osg::Vec3(0,1,0));
        osg::Quat _qYaw(osg::DegreesToRadians(_yaw), osg::Vec3(0,0,1));
        osg::Quat _modelRot = _qYaw * _qPitch * _qRoll;

        osg::Matrix _mat = osg::Matrix::rotate(_modelRot) * osg::Matrix::translate(-_posY, -_posX, _posZ);
        _planeTransform->setMatrix(_mat);
    }

    // --- Getters / Setters ---
    float& roll() { return _roll; }
    float& pitch() { return _pitch; }
    float& yaw() { return _yaw; }
    float& posX() { return _posX; }
    float& posY() { return _posY; }
    float& posZ() { return _posZ; }

    bool& circleEnabled() { return _circleEnabled; }
    float& circleRadius() { return _circleRadius; }
    float& circleSpeed() { return _circleSpeed; }

    public:
    osg::ref_ptr<osg::MatrixTransform> getTransform() const { return _planeTransform; }

    void reset()
    {
        _roll = _pitch = _yaw = 0.0f;
        _posX = _posY = _posZ = 0.0f;
        _angle = 0.0f;
    }

private:
    osg::ref_ptr<osg::MatrixTransform> _planeTransform;

    // Rotation
    float _roll, _pitch, _yaw;

    // Position
    float _posX, _posY, _posZ;

    // Circular trajectory
    bool _circleEnabled;
    float _circleRadius;
    float _circleSpeed;
    float _angle;
};

// ---------------- ImGuiPlaneHandler Class ----------------
class ImGuiPlaneHandler : public OsgImGuiHandler
{
public:
    ImGuiPlaneHandler(PlaneMotion* _motion, osgViewer::Viewer* _viewer)
        : _motion(_motion), _viewer(_viewer), _currentView(CameraView::Chase)
    {
        setupManipulator(_currentView);
    }

    void drawUi() override
    {
        ImGui::Begin("Plane Controls");

        // Camera view selection
        const char* _views[] = {"Front", "Chase", "Top"};
        static int _selectedView = 0;
        if (ImGui::Combo("Camera View", &_selectedView, _views, IM_ARRAYSIZE(_views)))
        {
            _currentView = static_cast<CameraView>(_selectedView);
            setupManipulator(_currentView);
        }

        ImGui::Separator();

        // Roll / Pitch / Yaw
        ImGui::Text("Roll"); ImGui::SliderFloat("##RollSlider", &_motion->roll(), -180.0f, 180.0f); ImGui::SameLine(); ImGui::InputFloat("##RollInput", &_motion->roll(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Pitch"); ImGui::SliderFloat("##PitchSlider", &_motion->pitch(), -180.0f, 180.0f); ImGui::SameLine(); ImGui::InputFloat("##PitchInput", &_motion->pitch(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Yaw"); ImGui::SliderFloat("##YawSlider", &_motion->yaw(), -180.0f, 180.0f); ImGui::SameLine(); ImGui::InputFloat("##YawInput", &_motion->yaw(), 1.0f, 10.0f, "%.1f");

        ImGui::Separator();

        // Position sliders
        ImGui::Text("Left/Right (X)"); ImGui::SliderFloat("##PosXSlider", &_motion->posX(), -100.0f, 100.0f); ImGui::SameLine(); ImGui::InputFloat("##PosXInput", &_motion->posX(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Forward/Backward (Y)"); ImGui::SliderFloat("##PosYSlider", &_motion->posY(), -100.0f, 100.0f); ImGui::SameLine(); ImGui::InputFloat("##PosYInput", &_motion->posY(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Up/Down (Z)"); ImGui::SliderFloat("##PosZSlider", &_motion->posZ(), -50.0f, 50.0f); ImGui::SameLine(); ImGui::InputFloat("##PosZInput", &_motion->posZ(), 1.0f, 10.0f, "%.1f");

        ImGui::Separator();

        // Circular trajectory
        ImGui::Checkbox("Enable Circular Trajectory", &_motion->circleEnabled());
        ImGui::SliderFloat("Radius", &_motion->circleRadius(), 5.0f, 100.0f);
        ImGui::SliderFloat("Speed", &_motion->circleSpeed(), 0.1f, 5.0f);

        ImGui::Separator();

        if (ImGui::Button("Reset"))
            _motion->reset();

        ImGui::End();

        // Update motion
        _motion->update();
    }

private:
    void setupManipulator(CameraView _view)
    {
        osg::ref_ptr<osgGA::NodeTrackerManipulator> _manip = new osgGA::NodeTrackerManipulator;
        _manip->setTrackNode(_motion->getTransform());
        _manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
        _manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

        osg::Quat _yawFix(osg::DegreesToRadians(90.0), osg::Vec3(0,0,1));

        osg::Vec3d _eye, _center, _up;
        switch(_view)
        {
            case CameraView::Front: _eye= {0,-50,20}; _center={0,0,5}; _up={0,0,1}; _eye=_yawFix*_eye; _up=_yawFix*_up; break;
            case CameraView::Chase: _eye= {0,50,20}; _center={0,0,5}; _up={0,0,1}; _eye=_yawFix*_eye; _up=_yawFix*_up; break;
            case CameraView::Top: _eye= {0,0,150}; _center={0,0,0}; _up={0,1,0}; break;
        }

        _manip->setHomePosition(_eye, _center, _up);
        _viewer->setCameraManipulator(_manip, true);
        _manip->home(0.0);
    }

    PlaneMotion* _motion;
    osgViewer::Viewer* _viewer;
    CameraView _currentView;
};

// ---------------- ImGui Init ----------------
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object*) override
    {
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ---------------- main ----------------
int main()
{
    osg::ref_ptr<osg::Group> _root = new osg::Group;
    std::string _dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> _fighterModel =
        osgDB::readRefNodeFile(_dataPath + "F-14-low-poly-osg-no-landgear.ac");

    if (!_fighterModel) return 1;

    osg::ref_ptr<osg::MatrixTransform> _fighterTransform = new osg::MatrixTransform;
    _fighterTransform->addChild(_fighterModel);
    _root->addChild(_fighterTransform);

    osgViewer::Viewer _viewer;
    _viewer.setSceneData(_root.get());
    _viewer.setUpViewInWindow(700, 50, 600, 600);

    _viewer.setRealizeOperation(new ImGuiInitOperation);

    PlaneMotion _motion(_fighterTransform);
    _viewer.addEventHandler(new ImGuiPlaneHandler(&_motion, &_viewer));

    return _viewer.run();
}
