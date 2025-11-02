#include <iostream>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/MatrixTransform>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= ImGui Initialization ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context) return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ======================= Global Control State ===========================
struct AnimationState
{
    bool running = false;
    float t = 0.0f;        // current normalized time [0, 1]
    float speed = 0.2f;    // motion speed
} gAnim;

// ======================= Trajectory Function ===========================
osg::Vec3 computeSTrajectory(float t)
{
    // "S" shaped trajectory in X-Z plane
    float x = (t - 0.5f) * 20.0f;
    float z = 5.0f * sinf(3.1415f * (t * 2.0f)); // sinusoidal "S"
    float y = 2.0f * cosf(3.1415f * (t * 2.0f));
    return osg::Vec3(x, y, z);
}

// ======================= Update Callback ===========================
class PlaneUpdateCallback : public osg::NodeCallback
{
public:
    osg::ref_ptr<osg::PositionAttitudeTransform> pat;

    PlaneUpdateCallback(osg::PositionAttitudeTransform* p) : pat(p) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        if (gAnim.running)
        {
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t > 1.0f) gAnim.t = 1.0f;
        }

        // compute current position
        osg::Vec3 pos = computeSTrajectory(gAnim.t);
        osg::Vec3 nextPos = computeSTrajectory(std::min(gAnim.t + 0.01f, 1.0f));

        osg::Vec3 dir = nextPos - pos;
        dir.normalize();

        osg::Quat attitude;
        attitude.makeRotate(osg::Vec3(1,0,0), dir);

        pat->setPosition(pos);
        pat->setAttitude(attitude);

        traverse(node, nv);
    }
};

// ======================= ImGui UI Handler ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl() {}

protected:
    void drawUi() override
    {
        ImGui::Begin("Plane Control");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
        }

        ImGui::SliderFloat("Progress", &gAnim.t, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");

        ImGui::End();
    }
};

// ======================= Utility: Draw S-curve Line ===========================
osg::ref_ptr<osg::Geode> createSTrajectoryLine()
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    for (float t = 0.0f; t <= 1.0f; t += 0.02f)
        vertices->push_back(computeSTrajectory(t));

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setVertexArray(vertices);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, vertices->size()));

    osg::ref_ptr<osg::Vec4Array> color = new osg::Vec4Array();
    color->push_back(osg::Vec4(1, 1, 0, 1));
    geom->setColorArray(color, osg::Array::BIND_OVERALL);

    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);
    geode->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);
    return geode;
}

// ======================= Utility: Create Plane Model ===========================
osg::ref_ptr<osg::PositionAttitudeTransform> createPlane()
{
    osg::ref_ptr<osg::PositionAttitudeTransform> pat = new osg::PositionAttitudeTransform();

    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(new osg::Box(osg::Vec3(), 1.0f, 0.3f, 0.1f));
    shape->setColor(osg::Vec4(0.2f, 0.7f, 1.0f, 1.0f));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(shape);

    pat->addChild(geode);
    pat->setPosition(computeSTrajectory(0.0f));

    return pat;
}

// ======================= Main ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    // Trajectory line
    root->addChild(createSTrajectoryLine());

    // Plane
    osg::ref_ptr<osg::PositionAttitudeTransform> plane = createPlane();
    root->addChild(plane);

    // Add plane motion callback
    plane->addUpdateCallback(new PlaneUpdateCallback(plane));

    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 800, 600));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.setLightingMode(osg::View::NO_LIGHT);

    return viewer.run();
}