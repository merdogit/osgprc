#include <iostream>
#include <algorithm>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osg/Math>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= Constants ===========================
const osg::Quat F14_BASIS(-0.00622421f, 0.713223f, -0.700883f, -0.0061165f);
const osg::Quat ROLL_180(osg::DegreesToRadians(180.0f), osg::Vec3(1, 0, 0));
// const osg::Quat MISSILE_BASIS(0.0f, 0.0f, 1.0f, 1.2168e-08f);
const osg::Quat MISSILE_BASIS(osg::PI, osg::Vec3(0,1,0));
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
    bool   running = false;
    bool   collided = false;
    float  t = 0.0f;
    float  speed = 0.25f;
    float  collisionThreshold = COLLISION_THRESHOLD_DEFAULT;
} gAnim;

// ======================= Forward declarations ===========================
struct TrajectoryCallback;
TrajectoryCallback* gAircraftTrail = nullptr;
TrajectoryCallback* gMissileTrail  = nullptr;

// ======================= Trajectories ===========================
static inline float easeCos01(float t)
{
    return 0.5f * (1.0f - cosf(osg::PI * std::clamp(t, 0.0f, 1.0f)));
}

// F-14 flight arc
osg::Vec3 aircraftTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float x = -60.0f + 120.0f * t;
    float y = 20.0f * easeCos01(t);
    float z = 5.0f + 8.0f * easeCos01(t);
    return osg::Vec3(x, y, z);
}

// Missile intercept arc
osg::Vec3 missileTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float x = 80.0f - 100.0f * t;
    float y = -15.0f * easeCos01(t);
    float z = 5.0f + 12.0f * easeCos01(t);
    return osg::Vec3(x, y, z);
}

// ======================= Frame Alignment Helper (X fwd, Y right, Z up) ===========================
static osg::Quat frameAlignLevel(const osg::Vec3 &forward_world, const osg::Vec3 &worldUp)
{
    osg::Vec3 Xw = forward_world;
    if (Xw.length2() < 1e-10f) Xw.set(1,0,0);
    Xw.normalize();

    // Keep wings level: project up onto plane orthogonal to forward
    osg::Vec3 Zw = worldUp - Xw * (worldUp * Xw);
    if (Zw.length2() < 1e-10f) Zw.set(0,0,1);
    Zw.normalize();

    osg::Vec3 Yw = Zw ^ Xw;
    if (Yw.length2() < 1e-10f) Yw.set(0,1,0);
    Yw.normalize();

    osg::Matrix R( Xw.x(), Yw.x(), Zw.x(), 0.0f,
                   Xw.y(), Yw.y(), Zw.y(), 0.0f,
                   Xw.z(), Yw.z(), Zw.z(), 0.0f,
                   0.0f,  0.0f,  0.0f,  1.0f );
    osg::Quat q; q.set(R);
    return q;
}

// ======================= Object Update Callback (fixed bank sign & composition order) ==============
class ObjectUpdateCallback : public osg::NodeCallback
{
public:
    osg::ref_ptr<osg::MatrixTransform> mt;
    bool isMissile;

    ObjectUpdateCallback(osg::MatrixTransform* m, bool missile)
        : mt(m), isMissile(missile) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        if (gAnim.running && !gAnim.collided) {
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t > 1.0f) gAnim.t = 1.0f;
        }

        auto traj = isMissile ? missileTrajectory : aircraftTrajectory;

        const float dt = 0.02f;
        const float t0 = std::max(0.0f, gAnim.t - dt);
        const float t2 = std::min(1.0f, gAnim.t + dt);

        const osg::Vec3 p0 = traj(t0);
        const osg::Vec3 p1 = traj(gAnim.t);
        const osg::Vec3 p2 = traj(t2);

        osg::Vec3 Tprev = p1 - p0;
        osg::Vec3 Tnext = p2 - p1;
        if (Tprev.length2() < 1e-12f) Tprev = Tnext;
        if (Tnext.length2() < 1e-12f) Tnext = Tprev;
        Tprev.normalize(); Tnext.normalize();

        const osg::Vec3 pos = p1;
        const osg::Vec3 worldUp(0,0,1);

        // base orientation: along path, wings level
        const osg::Vec3 fwd = Tnext;
        osg::Quat orientLevel = frameAlignLevel(fwd, worldUp);

        // ----- signed curvature & bank (left turn => left wing down) -----
        const float dotTN   = osg::clampBetween(Tprev * Tnext, -1.0f, 1.0f);
        const float ang     = acosf(dotTN);                      // [0,pi]
        const float sign    = ((Tprev ^ Tnext) * worldUp) >= 0 ? +1.0f : -1.0f;
        const float signedCurv = sign * ang;

        const float bankGain   = isMissile ? 1.2f : 1.8f;
        const float maxBankDeg = isMissile ? 35.0f : 55.0f;

        // negative roll for left turn with X-forward, Y-right, Z-up
        const float bankAngle = osg::clampBetween(-bankGain * signedCurv,
                                                  -osg::DegreesToRadians(maxBankDeg),
                                                   osg::DegreesToRadians(maxBankDeg));

        const osg::Quat bankLocal(bankAngle, osg::Vec3(1,0,0));    // roll about local +X
        const osg::Quat orient = orientLevel * bankLocal;          // bank in local frame

        // ----- compose with model basis (motion first, asset fix second) -----
        osg::Quat finalRot;
        if (isMissile) {
            finalRot = orient * MISSILE_BASIS;
        } else {
            finalRot = orient * (ROLL_180 * F14_BASIS);
            // If your F-14 model still noses 180° off, try:
            // finalRot = orient * (F14_BASIS * ROLL_180);
        }

        mt->setMatrix(osg::Matrix::rotate(finalRot) * osg::Matrix::translate(pos));

        // collision (unchanged)
        if (!isMissile) {
            const osg::Vec3 mpos = missileTrajectory(gAnim.t);
            if ((pos - mpos).length() < gAnim.collisionThreshold && !gAnim.collided) {
                gAnim.collided = true;
                gAnim.running = false;
                std::cout << "Collision at (" << pos.x() << ", " << pos.y() << ", " << pos.z() << ")\n";
            }
        }

        traverse(node, nv);
    }
};

// ======================= Trajectory Callback ===========================
struct TrajectoryCallback : public osg::NodeCallback
{
    osg::ref_ptr<osg::Vec3Array> vertices;
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::MatrixTransform> mt;

    TrajectoryCallback(osg::Geometry* g, osg::MatrixTransform* m, const osg::Vec4& color)
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

    void operator()(osg::Node*, osg::NodeVisitor*) override
    {
        osg::Vec3 pos = mt->getMatrix().getTrans();
        vertices->push_back(pos);
        auto* da = dynamic_cast<osg::DrawArrays*>(geom->getPrimitiveSet(0));
        if (da) da->setCount(vertices->size());
        geom->dirtyDisplayList();
        geom->dirtyBound();
    }

    void clearTrail()
    {
        vertices->clear();
        auto* da = dynamic_cast<osg::DrawArrays*>(geom->getPrimitiveSet(0));
        if (da) da->setCount(0);
        geom->dirtyDisplayList();
        geom->dirtyBound();
    }
};

// ======================= ImGui Control ===========================
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("F-14 vs AIM-9L Control");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
            gAnim.collided = false;
            if (gAircraftTrail) gAircraftTrail->clearTrail();
            if (gMissileTrail)  gMissileTrail->clearTrail();
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::SliderFloat("Collision Threshold", &gAnim.collisionThreshold, 0.5f, 5.0f, "%.2f");
        ImGui::Text("Collision: %s", gAnim.collided ? "YES" : "NO");

        ImGui::End();
    }
};

// ======================= Helpers ===========================
osg::ref_ptr<osg::Geode> createDynamicTrajectory(osg::MatrixTransform* mt, const osg::Vec4& color, bool isMissile)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    auto* cb = new TrajectoryCallback(geom, mt, color);
    mt->addUpdateCallback(cb);

    if (isMissile) gMissileTrail = cb;
    else           gAircraftTrail = cb;

    return geode;
}

// ======================= Main ===========================
int main(int, char**)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    // F-14
    osg::ref_ptr<osg::Node> f14 = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");
    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->addChild(f14);
    aircraft->addUpdateCallback(new ObjectUpdateCallback(aircraft, false));
    root->addChild(aircraft);

    // Missile
    osg::ref_ptr<osg::Node> missileModel = osgDB::readRefNodeFile(dataPath + "AIM-9L.ac");
    osg::ref_ptr<osg::MatrixTransform> missile = new osg::MatrixTransform;
    missile->addChild(missileModel);
    missile->addUpdateCallback(new ObjectUpdateCallback(missile, true));
    root->addChild(missile);

    // Trails
    root->addChild(createDynamicTrajectory(aircraft, osg::Vec4(0, 1, 0, 1), false));
    root->addChild(createDynamicTrajectory(missile,  osg::Vec4(1, 1, 0, 1), true));

    // osg::ref_ptr<osg::Node> refAxes = osgDB::readRefNodeFile(dataPath + "axes.osgt");
    // root->addChild(refAxes);

    // Viewer
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.setLightingMode(osg::View::SKY_LIGHT);

    return viewer.run();
}