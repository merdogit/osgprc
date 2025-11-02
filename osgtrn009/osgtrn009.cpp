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
    ImGuiPlaneHandler(osg::ref_ptr<osg::MatrixTransform> modelTransform,
                      osgViewer::Viewer* viewer)
        : planeTransform_(modelTransform), viewer_(viewer)
    {
        roll = pitch = yaw = 0.0f;
        currentView = CameraView::Chase;
        setupManipulator(currentView); // initialize manipulator
    }

    void drawUi() override
    {
        ImGui::Begin("Plane Controls");

        // Camera View selection
        const char* views[] = { "Front", "Chase", "Top" };
        static int selectedView = 0;
        if (ImGui::Combo("Camera View", &selectedView, views, IM_ARRAYSIZE(views)))
        {
            currentView = static_cast<CameraView>(selectedView);
            setupManipulator(currentView);
        }

        ImGui::Separator();

        // Roll
        ImGui::Text("Roll");
        ImGui::SliderFloat("##RollSlider", &roll, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##RollInput", &roll, 1.0f, 10.0f, "%.1f");

        // Pitch
        ImGui::Text("Pitch");
        ImGui::SliderFloat("##PitchSlider", &pitch, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##PitchInput", &pitch, 1.0f, 10.0f, "%.1f");

        // Yaw
        ImGui::Text("Yaw");
        ImGui::SliderFloat("##YawSlider", &yaw, -180.0f, 180.0f);
        ImGui::SameLine();
        ImGui::InputFloat("##YawInput", &yaw, 1.0f, 10.0f, "%.1f");

        if (ImGui::Button("Reset"))
        {
            roll = pitch = yaw = 0.0f;
        }

        ImGui::End();

        // Apply plane rotation
        osg::Quat qRoll(osg::DegreesToRadians(roll), osg::Vec3(1,0,0));
        osg::Quat qPitch(osg::DegreesToRadians(pitch), osg::Vec3(0,1,0));
        osg::Quat qYaw(osg::DegreesToRadians(yaw), osg::Vec3(0,0,1));

        osg::Quat modelRot = qYaw * qPitch * qRoll;
        planeTransform_->setMatrix(osg::Matrix::rotate(modelRot));
    }

private:
    void setupManipulator(CameraView view)
    {
        osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
        manip->setTrackNode(planeTransform_);
        manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
        manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

        osg::Quat yawFix(osg::DegreesToRadians(90.0), osg::Vec3(0,0,1));

        osg::Vec3d eye, center, up;
        switch(view)
        {
            case CameraView::Front:
                eye = osg::Vec3d(0.0, -50.0, 20.0);
                center = osg::Vec3d(0.0, 0.0, 5.0);
                up = osg::Vec3d(0.0, 0.0, 1.0);
                // Apply -90 yaw fix
                eye = yawFix * eye;
                up  = yawFix * up;
                break;
            case CameraView::Chase:
                eye = osg::Vec3d(0.0, 50.0, 20.0);
                center = osg::Vec3d(0.0, 0.0, 5.0);
                up = osg::Vec3d(0.0, 0.0, 1.0);
                // Apply -90 yaw fix
                eye = yawFix * eye;
                up  = yawFix * up;
                break;
            case CameraView::Top:
                eye = osg::Vec3d(0.0, 0.0, 150.0);
                center = osg::Vec3d(0.0, 0.0, 0.0);
                up = osg::Vec3d(0.0, 1.0, 0.0);
                // no fix needed
                break;
        }

        manip->setHomePosition(eye, center, up);

        viewer_->setCameraManipulator(manip, true);
        manip->home(0.0);
    }

    osg::ref_ptr<osg::MatrixTransform> planeTransform_;
    osgViewer::Viewer* viewer_;
    float roll, pitch, yaw;
    CameraView currentView;
};

// --- ImGui Init ---
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}

    void operator()(osg::Object* /*object*/) override
    {
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

int main()
{
    // Plane window
    osg::ref_ptr<osg::Group> root = new osg::Group;
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> fighterModel = 
    osgDB::readRefNodeFile(dataPath + "F-14-low-poly-osg-no-landgear.ac");
    
    if (!fighterModel)
        return 1;

    osg::ref_ptr<osg::MatrixTransform> fighterModelTransform = new osg::MatrixTransform;
    fighterModelTransform->addChild(fighterModel);
    root->addChild(fighterModelTransform);

    osgViewer::Viewer viewer;
    viewer.setSceneData(root.get());
    viewer.setUpViewInWindow(700, 50, 600, 600);

    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiPlaneHandler(fighterModelTransform, &viewer));

    return viewer.run();
}