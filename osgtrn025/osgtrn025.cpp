#include <iostream>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
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
    float t = 0.0f;        // [0, 1]
    float speed = 0.2f;    // speed factor
} gAnim;

// ======================= Trajectory Functions ===========================
// Missile from left to right
osg::Vec3 missileTrajectory(float t)
{
    float x = (t - 0.5f) * 20.0f; // -10 → +10
    float z = 5.0f * sinf(3.1415f * (t * 2.0f));
    float y = 2.0f * cosf(3.1415f * (t * 2.0f));
    return osg::Vec3(x, y, z);
}

// Aircraft from right to left (mirrored)
osg::Vec3 aircraftTrajectory(float t)
{
    float x = (0.5f - t) * 20.0f; // +10 → -10
    float z = -5.0f * sinf(3.1415f * (t * 2.0f));
    float y = 2.0f * cosf(3.1415f * (t * 2.0f));
    return osg::Vec3(x, y, z);
}

// ======================= Update Callback ===========================
class ObjectUpdateCallback : public osg::NodeCallback
{
public:
    osg::ref_ptr<osg::PositionAttitudeTransform> pat;
    bool isMissile;

    ObjectUpdateCallback(osg::PositionAttitudeTransform* p, bool missile)
        : pat(p), isMissile(missile) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        if (gAnim.running)
        {
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t > 1.0f) gAnim.t = 1.0f;
        }

        osg::Vec3 pos = isMissile ? missileTrajectory(gAnim.t) : aircraftTrajectory(gAnim.t);
        osg::Vec3 nextPos = isMissile
            ? missileTrajectory(std::min(gAnim.t + 0.01f, 1.0f))
            : aircraftTrajectory(std::min(gAnim.t + 0.01f, 1.0f));

        osg::Vec3 dir = nextPos - pos;
        if (dir.length2() < 1e-8f)
        {
            // too small to normalize safely
            pat->setPosition(pos);
            traverse(node, nv);
            return;
        }

        dir.normalize();
        osg::Quat attitude;
        attitude.makeRotate(osg::Vec3(1,0,0), dir);

        pat->setPosition(pos);
        pat->setAttitude(attitude);

        traverse(node, nv);
    }
};


// ======================= ImGui Control Panel ===========================
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("Missile vs Aircraft Control");

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

// ======================= Draw S-shaped Line ===========================
osg::ref_ptr<osg::Geode> createTrajectoryLine(bool missile)
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    for (float t = 0.0f; t <= 1.0f; t += 0.02f)
        vertices->push_back(missile ? missileTrajectory(t) : aircraftTrajectory(t));

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setVertexArray(vertices);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, vertices->size()));

    osg::ref_ptr<osg::Vec4Array> color = new osg::Vec4Array();
    color->push_back(missile ? osg::Vec4(1, 0, 0, 1) : osg::Vec4(0, 1, 1, 1));
    geom->setColorArray(color, osg::Array::BIND_OVERALL);

    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);
    geode->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);
    return geode;
}

// ======================= Utility: Create Box Model ===========================
osg::ref_ptr<osg::PositionAttitudeTransform> createBox(const osg::Vec4& color, const osg::Vec3& startPos, const osg::Vec3& size)
{
    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(new osg::Box(osg::Vec3(), size.x(), size.y(), size.z()));
    shape->setColor(color);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(shape);

    osg::ref_ptr<osg::PositionAttitudeTransform> pat = new osg::PositionAttitudeTransform();
    pat->addChild(geode);
    pat->setPosition(startPos);
    return pat;
}

// ======================= Main ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    // Trajectories
    root->addChild(createTrajectoryLine(true));   // missile (red)
    root->addChild(createTrajectoryLine(false));  // aircraft (cyan)

    // Missile
    osg::ref_ptr<osg::PositionAttitudeTransform> missile = createBox(
        osg::Vec4(1, 0.2f, 0.2f, 1), missileTrajectory(0.0f), osg::Vec3(1.0f, 0.3f, 0.3f));
    missile->addUpdateCallback(new ObjectUpdateCallback(missile, true));
    root->addChild(missile);

    // Aircraft
    osg::ref_ptr<osg::PositionAttitudeTransform> aircraft = createBox(
        osg::Vec4(0.2f, 0.8f, 1.0f, 1), aircraftTrajectory(0.0f), osg::Vec3(2.0f, 0.6f, 0.4f));
    aircraft->addUpdateCallback(new ObjectUpdateCallback(aircraft, false));
    root->addChild(aircraft);

    // Viewer setup
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.setLightingMode(osg::View::NO_LIGHT);

    return viewer.run();
}