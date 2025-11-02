#include <iostream>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osg/LineStipple>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= Constants ===========================
const osg::Quat MODEL_BASIS(-0.00622421, 0.713223, -0.700883, -0.0061165);
const osg::Quat ROLL_180(osg::DegreesToRadians(180.0f), osg::Vec3(1, 0, 0));
const float COLLISION_THRESHOLD_DEFAULT = 2.0f;

// ======================= ImGui Init ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        auto *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context) return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ======================= Global Animation State ===========================
struct AnimationState
{
    bool running = false;
    bool collided = false;
    float t = 0.0f;
    float speed = 0.25f;
    float collisionThreshold = COLLISION_THRESHOLD_DEFAULT;
    double lastUpdateTime = 0.0;
} gAnim;

// ======================= Forward declarations ===========================
struct TrajectoryCallback;
TrajectoryCallback* gAircraftTrail = nullptr;
TrajectoryCallback* gMissileTrail  = nullptr;


// ======================= Trajectories ===========================
// F-14 flies along +X (left → right)
osg::Vec3 aircraftTrajectory(float t)
{
    float x = -60.0f + 120.0f * t;
    float y = 0.0f;
    float z = 5.0f;
    return osg::Vec3(x, y, z);
}

// Missile flies opposite (+X → -X)
osg::Vec3 missileTrajectory(float t)
{
    float x = 80.0f - 100.0f * t;
    float y = 0.0f;
    float z = 5.0f;
    return osg::Vec3(x, y, z);
}

// ======================= Frame Alignment Helper ===========================
static osg::Quat frameAlignQuat(const osg::Vec3 &forward_world, const osg::Vec3 &up_world)
{
    osg::Vec3 Xw = forward_world;
    if (Xw.length2() < 1e-10f)
        Xw.set(1, 0, 0);
    Xw.normalize();

    osg::Vec3 Zw = up_world;
    if (Zw.length2() < 1e-10f)
        Zw.set(0, 0, 1);

    if (fabs(Xw * Zw) > 0.999f)
        Zw = osg::Vec3(0, 1, 0);

    osg::Vec3 Yw = Zw ^ Xw;
    if (Yw.length2() < 1e-10f)
        Yw.set(0, 1, 0);
    Yw.normalize();
    Zw = Xw ^ Yw;
    Zw.normalize();

    const osg::Vec3 Xl(1, 0, 0); // local forward (+X)
    const osg::Vec3 Zl(0, 0, 1); // local bottom (+Z)
    osg::Vec3 Yl = Zl ^ Xl;
    Yl.normalize();

    osg::Matrix toW(Xw.x(), Yw.x(), Zw.x(), 0.0f,
                    Xw.y(), Yw.y(), Zw.y(), 0.0f,
                    Xw.z(), Yw.z(), Zw.z(), 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);

    osg::Matrix fromL(Xl.x(), Yl.x(), Zl.x(), 0.0f,
                      Xl.y(), Yl.y(), Zl.y(), 0.0f,
                      Xl.z(), Yl.z(), Zl.z(), 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f);

    osg::Matrix fromL_T;
    fromL_T.transpose(fromL);
    osg::Matrix R = toW * fromL_T;

    osg::Quat q;
    q.set(R);
    return q;
}

// ======================= Object Update Callback ===========================
class ObjectUpdateCallback : public osg::NodeCallback
{
public:
    osg::ref_ptr<osg::MatrixTransform> mt;
    bool isMissile;

    ObjectUpdateCallback(osg::MatrixTransform *m, bool missile)
        : mt(m), isMissile(missile) {}

    void operator()(osg::Node *node, osg::NodeVisitor *nv) override
    {
        if (gAnim.running && !gAnim.collided)
        {
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t >= 1.0f)
                gAnim.t = 1.0f;
        }

        osg::Vec3 pos = isMissile ? missileTrajectory(gAnim.t) : aircraftTrajectory(gAnim.t);
        osg::Vec3 nextPos = isMissile ? missileTrajectory(std::min(gAnim.t + 0.01f, 1.0f)) : aircraftTrajectory(std::min(gAnim.t + 0.01f, 1.0f));

        osg::Vec3 fwd = nextPos - pos;
        if (fwd.length2() < 1e-10f)
            fwd.set(1, 0, 0);
        fwd.normalize();

        const osg::Vec3 worldUp(0, 0, 1);
        osg::Quat orient = frameAlignQuat(fwd, worldUp);

        // === Apply model’s base orientation ===
        // === Apply model’s base orientation ===
        // F-14 model base rotation
        osg::Quat modelBasis(-0.00622421, 0.713223, -0.700883, -0.0061165);

        // Additional 180° roll around forward (+X) to flip belly downward
        osg::Quat roll180(osg::DegreesToRadians(180.0f), osg::Vec3(1, 0, 0));

        // Compose: first apply roll, then model basis, then dynamic orientation
        osg::Quat finalRot = roll180 * modelBasis * orient;

        osg::Matrix M = osg::Matrix::rotate(finalRot) * osg::Matrix::translate(pos);
        mt->setMatrix(M);

        // Simple collision detection
        if (!isMissile)
        {
            osg::Vec3 mpos = missileTrajectory(gAnim.t);
            if ((pos - mpos).length() < 2.0f && !gAnim.collided)
            {
                gAnim.collided = true;
                gAnim.running = false;
                std::cout << "Collision detected at ("
                          << pos.x() << ", "
                          << pos.y() << ", "
                          << pos.z() << ")"
                          << std::endl;
            }
        }

        traverse(node, nv);
    }
};

// ======================= Dynamic Trajectory Callback ===========================
struct TrajectoryCallback : public osg::NodeCallback
{
    osg::ref_ptr<osg::Vec3Array> vertices;
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::MatrixTransform> mt;

    TrajectoryCallback(osg::Geometry *g, osg::MatrixTransform *m, const osg::Vec4 &color)
        : geom(g), mt(m)
    {
        vertices = new osg::Vec3Array();
        geom->setVertexArray(vertices);
        geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 0));

        osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
        colors->push_back(color);
        geom->setColorArray(colors, osg::Array::BIND_OVERALL);

        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
        geom->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);
        geom->setUseDisplayList(false);
    }

    void operator()(osg::Node *node, osg::NodeVisitor *nv) override
    {
        osg::Vec3 pos = mt->getMatrix().getTrans();
        vertices->push_back(pos);

        osg::DrawArrays *da = dynamic_cast<osg::DrawArrays *>(geom->getPrimitiveSet(0));
        if (da)
            da->setCount(vertices->size());

        geom->dirtyDisplayList();
        geom->dirtyBound();
        traverse(node, nv);
    }

    void clearTrail()
    {
        vertices->clear();
        osg::DrawArrays *da = dynamic_cast<osg::DrawArrays *>(geom->getPrimitiveSet(0));
        if (da)
            da->setCount(0);
        geom->dirtyDisplayList();
        geom->dirtyBound();
    }
};

// ======================= ImGui Control Panel ===========================
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("F14 vs Missile Control");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
            gAnim.collided = false;
            if (gAircraftTrail)
                gAircraftTrail->clearTrail();
            if (gMissileTrail)
                gMissileTrail->clearTrail();
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::Text("Collision: %s", gAnim.collided ? "YES" : "NO");
        ImGui::End();
    }
};

// ======================= Helpers ===========================
osg::ref_ptr<osg::MatrixTransform> createBox(
    const osg::Vec4 &color, const osg::Vec3 &pos, const osg::Vec3 &size)
{
    osg::ref_ptr<osg::ShapeDrawable> shape =
        new osg::ShapeDrawable(new osg::Box(osg::Vec3(), size.x(), size.y(), size.z()));
    shape->setColor(color);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(shape);

    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform();
    mt->addChild(geode);
    mt->setMatrix(osg::Matrix::translate(pos));
    return mt;
}

osg::ref_ptr<osg::Geode> createDynamicTrajectory(
    osg::MatrixTransform *mt, const osg::Vec4 &color, bool isMissile)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    auto *cb = new TrajectoryCallback(geom, mt, color);
    mt->addUpdateCallback(cb);

    if (isMissile)
        gMissileTrail = cb;
    else
        gAircraftTrail = cb;

    return geode;
}

// ======================= Main ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    // Load F-14 model
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> fighterModel =
        osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");

    // F-14 setup
    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->addChild(fighterModel);
    aircraft->addUpdateCallback(new ObjectUpdateCallback(aircraft, false));
    root->addChild(aircraft);

    // Missile (red box)
    osg::ref_ptr<osg::MatrixTransform> missile = createBox(
        osg::Vec4(1, 0.2f, 0.2f, 1),
        missileTrajectory(0.0f),
        osg::Vec3(1.0f, 0.3f, 0.3f));
    missile->addUpdateCallback(new ObjectUpdateCallback(missile, true));
    root->addChild(missile);

    // Trails
    root->addChild(createDynamicTrajectory(aircraft, osg::Vec4(0, 1, 0, 1), false)); // green
    root->addChild(createDynamicTrajectory(missile, osg::Vec4(1, 1, 0, 1), true));   // yellow

    // Viewer
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.setLightingMode(osg::View::SKY_LIGHT);

    return viewer.run();
}