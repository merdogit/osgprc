#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgGA/GUIEventHandler>

#include "/home/murate/Documents/SwTrn/DearIamGuiTrn/imgui/imgui.h"
#include "/home/murate/Documents/SwTrn/DearIamGuiTrn/imgui/backends/imgui_impl_opengl3.h"
#include "/home/murate/Documents/SwTrn/DearIamGuiTrn/imgui/backends/imgui_impl_glfw.h"

#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <filesystem>

// --- Helper to create axes (same as before)
osg::ref_ptr<osg::MatrixTransform> createAxis(
    const osg::Vec3 &axisDir, const osg::Vec4 &color, const std::string &label,
    float length = 5.0f, float radius = 0.1f, float coneRadius = 0.2f, float coneHeight = 0.5f)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    osg::ref_ptr<osg::Cylinder> bar = new osg::Cylinder(osg::Vec3(0,0,length*0.5f), radius, length);
    osg::ref_ptr<osg::ShapeDrawable> barDrawable = new osg::ShapeDrawable(bar);
    barDrawable->setColor(color);
    geode->addDrawable(barDrawable);

    osg::ref_ptr<osg::Cone> arrow = new osg::Cone(osg::Vec3(0,0,length), coneRadius, coneHeight);
    osg::ref_ptr<osg::ShapeDrawable> arrowDrawable = new osg::ShapeDrawable(arrow);
    arrowDrawable->setColor(color);
    geode->addDrawable(arrowDrawable);

    osg::ref_ptr<osgText::Text> text = new osgText::Text;
    text->setFont("arial.ttf");
    text->setCharacterSize(0.7f);
    text->setAxisAlignment(osgText::Text::SCREEN);
    text->setPosition(osg::Vec3(0,0,length+coneHeight+0.2f));
    text->setText(label);
    text->setColor(color);
    geode->addDrawable(text);

    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
    osg::Quat rot;
    rot.makeRotate(osg::Vec3(0,0,1), axisDir);
    mt->setMatrix(osg::Matrix::rotate(rot));
    mt->addChild(geode);

    return mt;
}

osg::ref_ptr<osg::Group> createAxes(bool ned = false, bool bodyFrame = false)
{
    osg::ref_ptr<osg::Group> axes = new osg::Group;

    osg::ref_ptr<osg::Geode> originGeode = new osg::Geode;
    osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0,0,0), 0.3f);
    osg::ref_ptr<osg::ShapeDrawable> sphereDrawable = new osg::ShapeDrawable(sphere);
    sphereDrawable->setColor(osg::Vec4(1,1,1,1));
    originGeode->addDrawable(sphereDrawable);
    axes->addChild(originGeode);

    if (!ned) {
        axes->addChild(createAxis(osg::Vec3(1,0,0), osg::Vec4(1,0,0,1), "X"));
        axes->addChild(createAxis(osg::Vec3(0,1,0), osg::Vec4(0,1,0,1), "Y"));
        axes->addChild(createAxis(osg::Vec3(0,0,1), osg::Vec4(0,0,1,1), "Z"));
    } else {
        if (bodyFrame) {
            axes->addChild(createAxis(osg::Vec3(-1,0,0), osg::Vec4(1,0,0,1), "BodyX")); // nose -> North
            axes->addChild(createAxis(osg::Vec3(0,1,0), osg::Vec4(0,1,0,1), "BodyY"));  // right wing -> East
            axes->addChild(createAxis(osg::Vec3(0,0,-1), osg::Vec4(0,0,1,1), "BodyZ")); // bottom -> Down
        } else {
            axes->addChild(createAxis(osg::Vec3(-1,0,0), osg::Vec4(1,0,0,1), "N"));
            axes->addChild(createAxis(osg::Vec3(0,1,0), osg::Vec4(0,1,0,1), "E"));
            axes->addChild(createAxis(osg::Vec3(0,0,-1), osg::Vec4(0,0,1,1), "D"));
        }
    }

    return axes;
}

// --- Fighter control handler
class FighterControlHandler : public osgGA::GUIEventHandler
{
public:
    FighterControlHandler(osg::ref_ptr<osg::MatrixTransform> fighterModel,
                          osg::ref_ptr<osg::MatrixTransform> fighterAxes)
        : _fighter(fighterModel), _axes(fighterAxes), _pitch(0), _yaw(0), _roll(0)
    {}

    virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&) override
    {
        if (!_fighter) return false;

        switch(ea.getEventType())
        {
        case osgGA::GUIEventAdapter::KEYDOWN:
            switch(ea.getKey())
            {
                case 'q': _pitch += 0.05; break;
                case 'a': _pitch -= 0.05; break;
                case 'w': _yaw   += 0.05; break;
                case 's': _yaw   -= 0.05; break;
                case 'e': _roll  += 0.05; break;
                case 'd': _roll  -= 0.05; break;
                case 'r': resetRotation(); break;
                default: return false;
            }
            updateRotation();
            return true;
        default: return false;
        }
    }

private:
    void updateRotation()
    {
        osg::Quat qPitch(_pitch, osg::Vec3(1,0,0));
        osg::Quat qYaw(_yaw, osg::Vec3(0,0,1));
        osg::Quat qRoll(_roll, osg::Vec3(0,1,0));
        osg::Quat finalRot = qYaw * qPitch * qRoll;

        _fighter->setMatrix(osg::Matrix::rotate(finalRot));
        if (_axes) _axes->setMatrix(osg::Matrix::rotate(finalRot));
    }

    void resetRotation()
    {
        _pitch = _yaw = _roll = 0.0;
        updateRotation();
    }

    osg::ref_ptr<osg::MatrixTransform> _fighter;
    osg::ref_ptr<osg::MatrixTransform> _axes;
    double _pitch, _yaw, _roll;
};

// --- Main ---
int main()
{
    // -----------------------
    // Axes window
    // -----------------------
    osg::ref_ptr<osg::Group> root1 = new osg::Group;
    root1->addChild(createAxes(true, false)); // Reference NED

    osg::ref_ptr<osg::MatrixTransform> fighterAxesTransform = new osg::MatrixTransform;
    fighterAxesTransform->addChild(createAxes(true, true)); // Body frame
    root1->addChild(fighterAxesTransform);

    osgViewer::Viewer viewer1;
    viewer1.setSceneData(root1.get());
    viewer1.setUpViewInWindow(50,50,600,600);
    viewer1.getCamera()->setViewMatrixAsLookAt(osg::Vec3(20,20,20), osg::Vec3(0,0,0), osg::Vec3(0,0,1));

    // -----------------------
    // Fighter model window
    // -----------------------
    osg::ref_ptr<osg::Group> root2 = new osg::Group;
    osg::ref_ptr<osg::MatrixTransform> fighterModelTransform = new osg::MatrixTransform;
    root2->addChild(fighterModelTransform);

    osgViewer::Viewer viewer2;
    viewer2.setSceneData(root2.get());
    viewer2.setUpViewInWindow(700,50,600,600);
    viewer2.setCameraManipulator(new osgGA::TrackballManipulator);
    viewer2.home();

    // -----------------------
    // Wait until context exists before ImGui init
    // -----------------------
    viewer2.realize(); // creates OpenGL context

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplOpenGL3_Init("#version 130");

    // Fighter keyboard
    viewer2.addEventHandler(new FighterControlHandler(fighterModelTransform, fighterAxesTransform));

    // -----------------------
    // Main loop
    // -----------------------
    std::string modelPath;
    while (!viewer1.done() && !viewer2.done())
    {
        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui::NewFrame();

        // Example window
        ImGui::Begin("Load Fighter Model");
        static char pathBuffer[512] = "";
        ImGui::InputText("Model Path", pathBuffer, sizeof(pathBuffer));
        if (ImGui::Button("Load"))
        {
            std::string path = pathBuffer;
            osg::ref_ptr<osg::Node> node = osgDB::readRefNodeFile(path);
            if (node)
            {
                fighterModelTransform->removeChildren(0,fighterModelTransform->getNumChildren());
                fighterModelTransform->addChild(node);

                // Optional: adjust orientation to NED here
                osg::Quat rot;
                rot.makeRotate(osg::Vec3(1,0,0), osg::Vec3(-1,0,0)); // example rotation
                fighterModelTransform->setMatrix(osg::Matrix::rotate(rot));
            }
        }
        ImGui::End();

        ImGui::Render();
        viewer2.frame();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        viewer1.frame();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();

    return 0;
}
