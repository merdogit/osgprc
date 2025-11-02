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

// --- Helper to create one axis ---
osg::ref_ptr<osg::MatrixTransform> createAxis(
    const osg::Vec3 &axisDir,
    const osg::Vec4 &color,
    const std::string &label,
    float length = 5.0f,
    float radius = 0.1f,
    float coneRadius = 0.2f,
    float coneHeight = 0.5f)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;

    osg::ref_ptr<osg::Cylinder> bar =
        new osg::Cylinder(osg::Vec3(0, 0, length * 0.5f), radius, length);
    osg::ref_ptr<osg::ShapeDrawable> barDrawable = new osg::ShapeDrawable(bar);
    barDrawable->setColor(color);
    geode->addDrawable(barDrawable);

    osg::ref_ptr<osg::Cone> arrow =
        new osg::Cone(osg::Vec3(0, 0, length), coneRadius, coneHeight);
    osg::ref_ptr<osg::ShapeDrawable> arrowDrawable = new osg::ShapeDrawable(arrow);
    arrowDrawable->setColor(color);
    geode->addDrawable(arrowDrawable);

    osg::ref_ptr<osgText::Text> text = new osgText::Text;
    text->setFont("arial.ttf");
    text->setCharacterSize(0.7f);
    text->setAxisAlignment(osgText::Text::SCREEN);
    text->setPosition(osg::Vec3(0, 0, length + coneHeight + 0.2f));
    text->setText(label);
    text->setColor(color);
    geode->addDrawable(text);

    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
    osg::Quat rot;
    rot.makeRotate(osg::Vec3(0, 0, 1), axisDir);
    mt->setMatrix(osg::Matrix::rotate(rot));
    mt->addChild(geode);

    return mt;
}

// --- Helper: create full XYZ triad ---
osg::ref_ptr<osg::Group> createAxes(const std::string &prefix = "",
                                    bool ned = false,
                                    bool bodyFrame = false)
{
    osg::ref_ptr<osg::Group> axes = new osg::Group;

    osg::ref_ptr<osg::Geode> originGeode = new osg::Geode;
    osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.3f);
    osg::ref_ptr<osg::ShapeDrawable> sphereDrawable = new osg::ShapeDrawable(sphere);
    sphereDrawable->setColor(osg::Vec4(1, 1, 1, 1));
    originGeode->addDrawable(sphereDrawable);
    axes->addChild(originGeode);

    if (!ned)
    {
        axes->addChild(createAxis(osg::Vec3(1, 0, 0), osg::Vec4(1, 0, 0, 1), prefix + "X"));
        axes->addChild(createAxis(osg::Vec3(0, 1, 0), osg::Vec4(0, 1, 0, 1), prefix + "Y"));
        axes->addChild(createAxis(osg::Vec3(0, 0, 1), osg::Vec4(0, 0, 1, 1), prefix + "Z"));
    }
    else
    {
        if (bodyFrame)
        {
            axes->addChild(createAxis(osg::Vec3(-1, 0, 0), osg::Vec4(1, 0, 0, 1), prefix + "X"));
            axes->addChild(createAxis(osg::Vec3(0, 1, 0), osg::Vec4(0, 1, 0, 1), prefix + "Y"));
            axes->addChild(createAxis(osg::Vec3(0, 0, -1), osg::Vec4(0, 0, 1, 1), prefix + "Z"));
        }
        else
        {
            axes->addChild(createAxis(osg::Vec3(-1, 0, 0), osg::Vec4(1, 0, 0, 1), prefix + "N"));
            axes->addChild(createAxis(osg::Vec3(0, 1, 0), osg::Vec4(0, 1, 0, 1), prefix + "E"));
            axes->addChild(createAxis(osg::Vec3(0, 0, -1), osg::Vec4(0, 0, 1, 1), prefix + "D"));
        }
    }

    return axes;
}

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
                      osg::ref_ptr<osg::MatrixTransform> axesTransform,
                      osgViewer::Viewer* viewer)
        : planeTransform_(modelTransform), axesTransform_(axesTransform), viewer_(viewer)
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

        if (axesTransform_)
            axesTransform_->setMatrix(osg::Matrix::rotate(modelRot));
    }

private:
    void setupManipulator(CameraView view)
    {
        osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
        manip->setTrackNode(planeTransform_);
        manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
        manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

        osg::Quat yawFix(osg::DegreesToRadians(-90.0), osg::Vec3(0,0,1));

        osg::Vec3d eye, center, up;
        switch(view)
        {
            case CameraView::Chase:
                eye = osg::Vec3d(0.0, -50.0, 20.0);
                center = osg::Vec3d(0.0, 0.0, 5.0);
                up = osg::Vec3d(0.0, 0.0, 1.0);
                // Apply -90 yaw fix
                eye = yawFix * eye;
                up  = yawFix * up;
                break;
            case CameraView::Front:
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
    osg::ref_ptr<osg::MatrixTransform> axesTransform_;
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
    // Axes window
    osg::ref_ptr<osg::Group> root1 = new osg::Group;
    root1->addChild(createAxes("", true));
    osg::ref_ptr<osg::MatrixTransform> fighterAxesTransform = new osg::MatrixTransform;
    fighterAxesTransform->addChild(createAxes("B", true, true));
    root1->addChild(fighterAxesTransform);

    osgViewer::Viewer viewer1;
    viewer1.setSceneData(root1.get());
    viewer1.setUpViewInWindow(50, 50, 600, 600);
    viewer1.getCamera()->setViewMatrixAsLookAt(
        osg::Vec3(20,20,20), osg::Vec3(0,0,0), osg::Vec3(0,0,1)
    );

    // Plane window
    osg::ref_ptr<osg::Group> root2 = new osg::Group;
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> fighterModel = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-osg.ac");
    if (!fighterModel)
        return 1;

    osg::ref_ptr<osg::MatrixTransform> fighterModelTransform = new osg::MatrixTransform;
    fighterModelTransform->addChild(fighterModel);
    root2->addChild(fighterModelTransform);

    osgViewer::Viewer viewer2;
    viewer2.setSceneData(root2.get());
    viewer2.setUpViewInWindow(700, 50, 600, 600);

    viewer2.setRealizeOperation(new ImGuiInitOperation);
    viewer2.addEventHandler(new ImGuiPlaneHandler(fighterModelTransform, fighterAxesTransform, &viewer2));

    // Main loop
    while (!viewer1.done() && !viewer2.done())
    {
        viewer1.frame();
        viewer2.frame();
    }

    return 0;
}