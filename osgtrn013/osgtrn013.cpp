#include <osg/Geometry>
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
    PlaneMotion(osg::ref_ptr<osg::MatrixTransform> planeTransform, osg::Group *root)
        : _planeTransform(planeTransform),
          _roll(0.0f), _pitch(0.0f), _yaw(0.0f),
          _posX(0.0f), _posY(0.0f), _posZ(0.0f),
          _circleEnabled(false), _circleRadius(20.0f),
          _circleSpeed(1.0f), _angle(0.0f),
          _root(root)
    {
        createReferenceCircle();
    }

    void update()
    {
        if (_circleEnabled)
        {
            // Update angle
            _angle += 0.01f * _circleSpeed;

            // Update position along circle
            _posX = _circleRadius * cos(_angle);
            _posY = _circleRadius * sin(_angle);

            // Update yaw and roll for circular motion
            _yaw = -_angle * 180.0f / 3.14159265f; // convert to degrees
            _roll = 20.0f * sin(_angle * 2.0f);    // banking effect
        }

        // Update reference circle dynamically
        updateCircleVertices();

        // Rotation
        osg::Quat qRoll(osg::DegreesToRadians(_roll), osg::Vec3(1, 0, 0));
        osg::Quat qPitch(osg::DegreesToRadians(_pitch), osg::Vec3(0, 1, 0));
        osg::Quat qYaw(osg::DegreesToRadians(_yaw), osg::Vec3(0, 0, 1));
        osg::Quat modelRot = qYaw * qPitch * qRoll;

        // Apply transformation
        osg::Matrix mat = osg::Matrix::rotate(modelRot) * osg::Matrix::translate(-_posY, -_posX, _posZ);
        _planeTransform->setMatrix(mat);
    }

    osg::ref_ptr<osg::MatrixTransform> getTransform() const { return _planeTransform; }

    // --- Getters / Setters ---
    float &roll() { return _roll; }
    float &pitch() { return _pitch; }
    float &yaw() { return _yaw; }
    float &posX() { return _posX; }
    float &posY() { return _posY; }
    float &posZ() { return _posZ; }

    bool &circleEnabled() { return _circleEnabled; }
    float &circleRadius() { return _circleRadius; }
    float &circleSpeed() { return _circleSpeed; }

    void reset()
    {
        _roll = _pitch = _yaw = 0.0f;
        _posX = _posY = _posZ = 0.0f;
        _angle = 0.0f;
    }

private:
    void createReferenceCircle()
    {
        osg::ref_ptr<osg::Geode> geode = new osg::Geode;

        _circleGeom = new osg::Geometry;
        _circleVertices = new osg::Vec3Array;
        _circleGeom->setVertexArray(_circleVertices);

        // Initial fill
        updateCircleVertices();

        _circleGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_LOOP, 0, _circleVertices->size()));

        // Color
        osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f)); // red
        _circleGeom->setColorArray(colors, osg::Array::BIND_OVERALL);

        geode->addDrawable(_circleGeom);
        _root->addChild(geode);
    }

    void updateCircleVertices()
    {
        if (!_circleVertices) return;

        _circleVertices->clear();
        const int segments = 64;
        for (int i = 0; i < segments; ++i)
        {
            float angle = osg::PI * 2.0f * i / segments;
            _circleVertices->push_back(osg::Vec3(
                cos(angle) * _circleRadius,
                sin(angle) * _circleRadius,
                0.0f));
        }
        _circleVertices->dirty();
        _circleGeom->dirtyDisplayList();
        _circleGeom->dirtyBound();
    }

    osg::ref_ptr<osg::MatrixTransform> _planeTransform;
    float _roll, _pitch, _yaw;
    float _posX, _posY, _posZ;

    bool _circleEnabled;
    float _circleRadius;
    float _circleSpeed;
    float _angle;

    osg::ref_ptr<osg::Geometry> _circleGeom;
    osg::ref_ptr<osg::Vec3Array> _circleVertices;

    osg::Group *_root;
};

// ---------------- CameraUpdater Class ----------------
class CameraUpdater : public osg::NodeCallback
{
public:
    CameraUpdater(osgGA::NodeTrackerManipulator* manip,
                  PlaneMotion* motion,
                  CameraView* currentView)
        : _manip(manip), _motion(motion), _currentView(currentView) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        osg::Vec3d eye, center, up;

        osg::Matrix mat = _motion->getTransform()->getMatrix();
        osg::Vec3d planePos = mat.getTrans();

        std::cout << "planePosX: " << planePos.x() << '\n';
        std::cout << "planePosY: " << planePos.y() << '\n';
        std::cout << "planePosZ: " << planePos.z() << '\n';


        osg::Quat yawFix(osg::DegreesToRadians(90.0), osg::Vec3(0, 0, 1));

        switch (*_currentView)
        {
        case CameraView::Front:
            eye = planePos + osg::Vec3d(0.0, -50.0, 20.0);;
            center = planePos + osg::Vec3d(0.0, 0.0, 5.0);
            up = osg::Vec3d(0.0, 0.0, 1.0);
            break;
        case CameraView::Chase:
            eye = planePos + osg::Vec3d(0.0, 50.0, 20.0);
            center = planePos + osg::Vec3d(0.0, 0.0, 5.0);
            up = osg::Vec3d(0.0, 0.0, 1.0);
            break;
        case CameraView::Top:
            eye = planePos + osg::Vec3d(0.0, 0.0, 150.0);
            center = planePos;
            up = osg::Vec3d(0.0, 1.0, 0.0);
            break;
        }

        _manip->setHomePosition(-eye, -center, up, false);
        _manip->home(0.0);

        traverse(node, nv);
    }

private:
    osg::observer_ptr<osgGA::NodeTrackerManipulator> _manip;
    PlaneMotion* _motion;
    CameraView* _currentView;
};

// ---------------- ImGuiPlaneHandler Class ----------------
class ImGuiPlaneHandler : public OsgImGuiHandler
{
public:
    ImGuiPlaneHandler(PlaneMotion *motion, osgViewer::Viewer *viewer, osg::Group* root)
        : _motion(motion), _viewer(viewer), _root(root), _currentView(CameraView::Chase)
    {
        setupManipulator(_currentView);
    }

    void drawUi() override
    {
        ImGui::Begin("Plane Controls");

        // Camera view selection
        const char *views[] = {"Front", "Chase", "Top"};
        static int selectedView = 0;
        if (ImGui::Combo("Camera View", &selectedView, views, IM_ARRAYSIZE(views)))
        {
            _currentView = static_cast<CameraView>(selectedView);
            setupManipulator(_currentView);
        }

        ImGui::Separator();

        // Roll / Pitch / Yaw
        ImGui::Text("Roll");
        ImGui::SliderFloat("##RollSlider", &_motion->roll(), -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##RollInput", &_motion->roll(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Pitch");
        ImGui::SliderFloat("##PitchSlider", &_motion->pitch(), -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PitchInput", &_motion->pitch(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Yaw");
        ImGui::SliderFloat("##YawSlider", &_motion->yaw(), -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##YawInput", &_motion->yaw(), 1.0f, 10.0f, "%.1f");

        ImGui::Separator();

        // Position sliders
        ImGui::Text("Left/Right (X)");
        ImGui::SliderFloat("##PosXSlider", &_motion->posX(), -100.0f, 100.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosXInput", &_motion->posX(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Forward/Backward (Y)");
        ImGui::SliderFloat("##PosYSlider", &_motion->posY(), -100.0f, 100.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosYInput", &_motion->posY(), 1.0f, 10.0f, "%.1f");
        ImGui::Text("Up/Down (Z)");
        ImGui::SliderFloat("##PosZSlider", &_motion->posZ(), -50.0f, 50.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PosZInput", &_motion->posZ(), 1.0f, 10.0f, "%.1f");

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
    void setupManipulator(CameraView view)
    {
        osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
        manip->setTrackNode(_motion->getTransform());
        manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
        manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

        // assign manipulator to viewer
        _viewer->setCameraManipulator(manip, false);

        // attach updater callback to root so it runs every frame
        _root->setUpdateCallback(new CameraUpdater(manip.get(), _motion, &_currentView));
    }

    PlaneMotion *_motion;
    osgViewer::Viewer *_viewer;
    osg::Group* _root;
    CameraView _currentView;
};

// ---------------- ImGui Init ----------------
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *) override
    {
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ---------------- main ----------------
int main()
{
    osg::ref_ptr<osg::Group> root = new osg::Group;
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> fighterModel =
        osgDB::readRefNodeFile(dataPath + "F-14-low-poly-osg-no-landgear.ac");

    if (!fighterModel)
        return 1;

    osg::ref_ptr<osg::MatrixTransform> fighterTransform = new osg::MatrixTransform;
    fighterTransform->addChild(fighterModel);
    root->addChild(fighterTransform);

    osgViewer::Viewer viewer;
    viewer.setSceneData(root.get());
    viewer.setUpViewInWindow(700, 50, 600, 600);

    viewer.setRealizeOperation(new ImGuiInitOperation);

    PlaneMotion motion(fighterTransform, root.get());
    viewer.addEventHandler(new ImGuiPlaneHandler(&motion, &viewer, root.get()));

    return viewer.run();
}