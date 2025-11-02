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
#include "OsgImGuiHandler.hpp" // Your wrapper to integrate ImGui with OSG

// --- Camera view enum ---
enum class CameraView
{
    Chase,
    Front,
    Top
};

// --- ImGui handler ---
class ImGuiPlaneHandler : public OsgImGuiHandler
{
public:
    ImGuiPlaneHandler(osg::ref_ptr<osg::MatrixTransform> _modelTransform,
                      osgViewer::Viewer *_viewer)
        : _planeTransform(_modelTransform), _viewer(_viewer)
    {
        _roll = _pitch = _yaw = 0.0f;
        _posX = _posY = _posZ = 0.0f; // initialize position
        _currentView = CameraView::Chase;
        setupManipulator(_currentView);
    }

    void drawUi() override
    {
        ImGui::Begin("Plane Controls");

        // Camera View selection
        const char *_views[] = {"Front", "Chase", "Top"};
        static int _selectedView = 0;
        if (ImGui::Combo("Camera View", &_selectedView, _views, IM_ARRAYSIZE(_views)))
        {
            _currentView = static_cast<CameraView>(_selectedView);
            setupManipulator(_currentView);
        }

        ImGui::Separator();

        // Roll
        ImGui::Text("Roll");
        ImGui::SliderFloat("##RollSlider", &_roll, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##RollInput", &_roll, 1.0f, 10.0f, "%.1f");

        // Pitch
        ImGui::Text("Pitch");
        ImGui::SliderFloat("##PitchSlider", &_pitch, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PitchInput", &_pitch, 1.0f, 10.0f, "%.1f");

        // Yaw
        ImGui::Text("Yaw");
        ImGui::SliderFloat("##YawSlider", &_yaw, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##YawInput", &_yaw, 1.0f, 10.0f, "%.1f");

        ImGui::Separator();
        ImGui::Text("Position");

        // Translation sliders + input fields
        ImGui::Text("Left / Right (X)");
        ImGui::SliderFloat("##PosXSlider", &_posX, -100.0f, 100.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosXInput", &_posX, 1.0f, 10.0f, "%.1f");

        ImGui::Text("Forward / Backward (Y)");
        ImGui::SliderFloat("##PosYSlider", &_posY, -100.0f, 100.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosYInput", &_posY, 1.0f, 10.0f, "%.1f");

        ImGui::Text("Up / Down (Z)");
        ImGui::SliderFloat("##PosZSlider", &_posZ, -50.0f, 50.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosZInput", &_posZ, 1.0f, 10.0f, "%.1f");

        ImGui::Separator();

        if (ImGui::Button("Reset"))
        {
            _roll = _pitch = _yaw = 0.0f;
            _posX = _posY = _posZ = 0.0f;
        }

        ImGui::End();

        // Apply rotation
        osg::Quat _qRoll(osg::DegreesToRadians(_roll), osg::Vec3(1, 0, 0));
        osg::Quat _qPitch(osg::DegreesToRadians(_pitch), osg::Vec3(0, 1, 0));
        osg::Quat _qYaw(osg::DegreesToRadians(_yaw), osg::Vec3(0, 0, 1));
        osg::Quat _modelRot = _qYaw * _qPitch * _qRoll;

        // Apply translation + rotation
        osg::Matrix _mat = osg::Matrix::rotate(_modelRot) * osg::Matrix::translate(-_posY, -_posX, _posZ);
        _planeTransform->setMatrix(_mat);
    }

private:
    void setupManipulator(CameraView _view)
    {
        osg::ref_ptr<osgGA::NodeTrackerManipulator> _manip = new osgGA::NodeTrackerManipulator;
        _manip->setTrackNode(_planeTransform);
        _manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
        _manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

        osg::Quat _yawFix(osg::DegreesToRadians(90.0), osg::Vec3(0, 0, 1));

        osg::Vec3d _eye, _center, _up;
        switch (_view)
        {
        case CameraView::Front:
            _eye = osg::Vec3d(0.0, -50.0, 20.0);
            _center = osg::Vec3d(0.0, 0.0, 5.0);
            _up = osg::Vec3d(0.0, 0.0, 1.0);
            _eye = _yawFix * _eye;
            _up = _yawFix * _up;
            break;
        case CameraView::Chase:
            _eye = osg::Vec3d(0.0, 50.0, 20.0);
            _center = osg::Vec3d(0.0, 0.0, 5.0);
            _up = osg::Vec3d(0.0, 0.0, 1.0);
            _eye = _yawFix * _eye;
            _up = _yawFix * _up;
            break;
        case CameraView::Top:
            _eye = osg::Vec3d(0.0, 0.0, 150.0);
            _center = osg::Vec3d(0.0, 0.0, 0.0);
            _up = osg::Vec3d(0.0, 1.0, 0.0);
            break;
        }

        _manip->setHomePosition(_eye, _center, _up);
        _viewer->setCameraManipulator(_manip, true);
        _manip->home(0.0);
    }

    osg::ref_ptr<osg::MatrixTransform> _planeTransform;
    osgViewer::Viewer *_viewer;
    float _roll, _pitch, _yaw;
    float _posX, _posY, _posZ; // translation variables
    CameraView _currentView;
};

// --- ImGui Init ---
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}

    void operator()(osg::Object * /*object*/) override
    {
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

int main()
{
    // Plane window
    osg::ref_ptr<osg::Group> _root = new osg::Group;
    std::string _dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> _fighterModel =
        osgDB::readRefNodeFile(_dataPath + "F-14-low-poly-osg-no-landgear.ac");

    if (!_fighterModel)
        return 1;

    osg::ref_ptr<osg::MatrixTransform> _fighterModelTransform = new osg::MatrixTransform;
    _fighterModelTransform->addChild(_fighterModel);
    _root->addChild(_fighterModelTransform);

    osgViewer::Viewer _viewer;
    _viewer.setSceneData(_root.get());
    _viewer.setUpViewInWindow(700, 50, 600, 600);

    _viewer.setRealizeOperation(new ImGuiInitOperation);
    _viewer.addEventHandler(new ImGuiPlaneHandler(_fighterModelTransform, &_viewer));

    return _viewer.run();
}