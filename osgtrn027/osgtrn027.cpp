#include <iostream>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= ImGui Init ===========================
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

// ======================= Global Animation State ===========================
struct AnimationState
{
    bool running = false;
    float t = 0.0f;    // normalized time [0,1]
    float speed = 0.25f;
} gAnim;

// ======================= Trajectories ===========================
// Aircraft: comes from +Y, curves to -X
osg::Vec3 aircraftTrajectory(float t)
{
    float x = -10.0f * t + 2.0f * sin(t * osg::PI);
    float y = 10.0f * (1.0f - t);
    float z = 0.0f;
    return osg::Vec3(x, y, z);
}

// Missile: comes from -Y, curves to -X
osg::Vec3 missileTrajectory(float t)
{
    float x = -10.0f * t - 2.0f * sin(t * osg::PI);
    float y = -10.0f * (1.0f - t);
    float z = 0.0f;
    return osg::Vec3(x, y, z);
}

// ======================= Object Update Callback ===========================
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
            if (gAnim.t >= 1.0f)
            {
                gAnim.t = 1.0f;
                gAnim.running = false;
            }
        }

        osg::Vec3 pos = isMissile ? missileTrajectory(gAnim.t) : aircraftTrajectory(gAnim.t);
        osg::Vec3 nextPos = isMissile ?
            missileTrajectory(std::min(gAnim.t + 0.01f, 1.0f)) :
            aircraftTrajectory(std::min(gAnim.t + 0.01f, 1.0f));

        osg::Vec3 dir = nextPos - pos;
        if (dir.length2() < 1e-8f)
        {
            pat->setPosition(pos);
            traverse(node, nv);
            return;
        }

        dir.normalize();
        osg::Quat rot;
        rot.makeRotate(osg::Vec3(1, 0, 0), dir);

        pat->setPosition(pos);
        pat->setAttitude(rot);

        traverse(node, nv);
    }
};

// ======================= ImGui Control Panel ===========================
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("Missile vs Aircraft Control (X-Y plane)");

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

// ======================= Dynamic Trajectory Callback ===========================
class TrajectoryCallback : public osg::NodeCallback
{
public:
    osg::ref_ptr<osg::Vec3Array> vertices;
    osg::ref_ptr<osg::Geometry> geom;

    TrajectoryCallback(osg::Geometry* g)
        : geom(g)
    {
        vertices = new osg::Vec3Array();
        geom->setVertexArray(vertices);
        geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 0));

        osg::ref_ptr<osg::Vec4Array> color = new osg::Vec4Array();
        color->push_back(osg::Vec4(1,1,0,1)); // yellow for example
        geom->setColorArray(color, osg::Array::BIND_OVERALL);

        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
        geom->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);
        geom->setUseDisplayList(false); // important for dynamic update
    }

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        osg::PositionAttitudeTransform* pat = dynamic_cast<osg::PositionAttitudeTransform*>(node);
        if (!pat) return;

        vertices->push_back(pat->getPosition());

        osg::DrawArrays* da = dynamic_cast<osg::DrawArrays*>(geom->getPrimitiveSet(0));
        if (da) da->setCount(vertices->size());

        geom->dirtyDisplayList();
        geom->dirtyBound();

        traverse(node, nv);
    }
};

// ======================= Create Object Boxes ===========================
osg::ref_ptr<osg::PositionAttitudeTransform> createBox(const osg::Vec4& color, const osg::Vec3& pos, const osg::Vec3& size)
{
    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(new osg::Box(osg::Vec3(), size.x(), size.y(), size.z()));
    shape->setColor(color);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(shape);

    osg::ref_ptr<osg::PositionAttitudeTransform> pat = new osg::PositionAttitudeTransform();
    pat->addChild(geode);
    pat->setPosition(pos);
    return pat;
}

// ======================= Create dynamic trajectory geode ===========================
osg::ref_ptr<osg::Geode> createDynamicTrajectory(osg::PositionAttitudeTransform* pat)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    // Trajectory callback attached to the box PAT
    pat->addUpdateCallback(new TrajectoryCallback(geom));

    return geode; // will add to root to remain in world coordinates
}

// ======================= Main ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    // Aircraft
    osg::ref_ptr<osg::PositionAttitudeTransform> aircraft = createBox(
        osg::Vec4(0.2f, 0.8f, 1.0f, 1), aircraftTrajectory(0.0f), osg::Vec3(2.0f, 0.6f, 0.4f));
    aircraft->addUpdateCallback(new ObjectUpdateCallback(aircraft, false));
    root->addChild(aircraft);

    // Missile
    osg::ref_ptr<osg::PositionAttitudeTransform> missile = createBox(
        osg::Vec4(1, 0.2f, 0.2f, 1), missileTrajectory(0.0f), osg::Vec3(1.0f, 0.3f, 0.3f));
    missile->addUpdateCallback(new ObjectUpdateCallback(missile, true));
    root->addChild(missile);

    // Dynamic Trajectories
    osg::ref_ptr<osg::Geode> aircraftLine = createDynamicTrajectory(aircraft);
    osg::ref_ptr<osg::Geode> missileLine   = createDynamicTrajectory(missile);
    root->addChild(aircraftLine);
    root->addChild(missileLine);

    // Viewer setup
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.setLightingMode(osg::View::NO_LIGHT);

    return viewer.run();
}