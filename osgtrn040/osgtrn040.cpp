#include <iostream>
#include <iomanip>
#include <algorithm>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osgGA/TrackballManipulator>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/LineStipple>
#include <osgDB/ReadFile>
#include <osg/Math>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/PolygonOffset>
#include <osg/Camera>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// =============== ANSI (trimmed) ===============
#define ANSI_RESET "\e[0;0m]"
#define ANSI_CYAN "\e[0;36m"

// =============== ImGui init ===============
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        auto *gc = dynamic_cast<osg::GraphicsContext *>(object);
        if (!gc)
            return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// =============== Global state ===============
struct AnimationState
{
    bool running = false;
    bool logging = false;
    float t = 0.0f;
    float speed = 0.25f;
    int cameraMode = 0; // 0=Free, 1=F14, 2=Missile
} gAnim;

float gTailOffset = -14.0f;

// IMPORTANT: Use world-up (OSG) = +Z
const osg::Vec3 WORLD_UP(0, 0, 1);

// Your model bases
const osg::Quat F14_BASIS(-0.00622421, 0.713223, -0.700883, -0.0061165);
const osg::Quat MISSILE_BASIS(0, 0, 1, 0);

// =============== Trajectories ===============
osg::Vec3 aircraftTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float x = -120.0f + 240.0f * t;
    float amplitude = 15.0f;
    float cycles = 1.5f;
    float y = amplitude * sinf(cycles * 2.0f * osg::PI * t);
    float z = amplitude * sinf(cycles * 2.0f * osg::PI * t);
    return osg::Vec3(x, y, z);
}

osg::Vec3 missileTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float x = -120.0f + 260.0f * t + 10.0f;
    float y = 25.0f * sinf(1.2f * osg::PI * t);
    float z = -5.0f * t;
    return osg::Vec3(x, y, z);
}

// =============== Orientation (NED body; world Z-up) ===============
static osg::Quat orientationFromTangent(const osg::Vec3 &forward, const osg::Vec3 &worldUp)
{
    // Body axes definition in your project:
    // +X = nose (forward), +Y = right wing, +Z = down (NED)
    osg::Vec3 X = forward;
    X.normalize();

    // wings-level: take worldUp component perpendicular to X, then NEGATE to get body +Z (down)
    osg::Vec3 Z = -(worldUp - X * (worldUp * X));
    Z.normalize();

    // Right-hand rule Y = Z x X (with Z "down" this yields right)
    osg::Vec3 Y = Z ^ X;
    Y.normalize();

    osg::Matrix R(
        X.x(), Y.x(), Z.x(), 0.0f,
        X.y(), Y.y(), Z.y(), 0.0f,
        X.z(), Y.z(), Z.z(), 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

    return R.getRotate();
}

// =============== Debug axes ===============
osg::ref_ptr<osg::Geode> createAxes(float len = 10.0f)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array();

    // X (red)
    v->push_back({0, 0, 0});
    v->push_back({-len, 0, 0});
    c->push_back({1, 0, 0, 1});
    c->push_back({1, 0, 0, 1});
    // Y (green)
    v->push_back({0, 0, 0});
    v->push_back({0, 0, -len});
    c->push_back({0, 1, 0, 1});
    c->push_back({0, 1, 0, 1});
    // Z (blue)
    v->push_back({0, 0, 0});
    v->push_back({0, -len, 0});
    c->push_back({0, 0, 1, 1});
    c->push_back({0, 0, 1, 1});

    geom->setVertexArray(v);
    geom->setColorArray(c, osg::Array::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, v->size()));
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
    geom->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);
    return geode;
}

// =============== Trail ===============
class Trail : public osg::Referenced
{
public:
    Trail(size_t maxPoints = 2000, float minSegment = 0.2f)
        : _maxPoints(maxPoints), _minSegment(minSegment)
    {
        _verts = new osg::Vec3Array;
        _geom = new osg::Geometry;
        _draw = new osg::DrawArrays(GL_LINE_STRIP, 0, 0);
        _geom->setVertexArray(_verts.get());
        _geom->addPrimitiveSet(_draw.get());
        osg::ref_ptr<osg::Vec4Array> col = new osg::Vec4Array;
        col->push_back(osg::Vec4(1.0f, 1.0f, 0.4f, 0.9f)); // yellow
        _geom->setColorArray(col, osg::Array::BIND_OVERALL);
        osg::StateSet *ss = _geom->getOrCreateStateSet();
        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(2.5f);
        ss->setAttributeAndModes(lw, osg::StateAttribute::ON);
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
        _geode = new osg::Geode;
        _geode->addDrawable(_geom.get());
    }
    osg::Geode *geode() const { return _geode.get(); }
    void clear()
    {
        _verts->clear();
        _draw->setCount(0);
        _geom->dirtyDisplayList();
        _geom->dirtyBound();
        _hasLast = false;
    }
    void addPoint(const osg::Vec3 &p)
    {
        if (_hasLast && (p - _last).length() < _minSegment)
            return;
        _verts->push_back(p);
        _last = p;
        _hasLast = true;
        if (_verts->size() > _maxPoints)
            _verts->erase(_verts->begin(), _verts->begin() + (_verts->size() - _maxPoints));
        _draw->setCount(_verts->size());
        _geom->dirtyDisplayList();
        _geom->dirtyBound();
    }

private:
    osg::ref_ptr<osg::Geode> _geode;
    osg::ref_ptr<osg::Geometry> _geom;
    osg::ref_ptr<osg::Vec3Array> _verts;
    osg::ref_ptr<osg::DrawArrays> _draw;
    size_t _maxPoints;
    float _minSegment;
    bool _hasLast = false;
    osg::Vec3 _last;
};

// =============== Motion callbacks ===============
class F14MotionCallback : public osg::NodeCallback
{
public:
    F14MotionCallback(osg::MatrixTransform *m, Trail *trail, float tailOffset = -24.0f)
        : mt(m), _trail(trail), _tailOffset(tailOffset) {}
    osg::Vec3 pos;
    osg::Quat rot; // world position/orientation per frame

    void operator()(osg::Node *, osg::NodeVisitor *nv) override
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
        const float dt = 0.02f;
        float t0 = std::max(0.0f, gAnim.t - dt);
        float t2 = std::min(1.0f, gAnim.t + dt);

        osg::Vec3 p0 = aircraftTrajectory(t0);
        osg::Vec3 p1 = aircraftTrajectory(gAnim.t);
        osg::Vec3 p2 = aircraftTrajectory(t2);

        osg::Vec3 fwd = p2 - p1;
        if (fwd.length2() < 1e-8f)
            fwd = p1 - p0;
        fwd.normalize();
        osg::Quat body = orientationFromTangent(fwd, WORLD_UP);
        rot = body * F14_BASIS;
        pos = p1;

        mt->setMatrix(osg::Matrix::rotate(rot) * osg::Matrix::translate(pos));

        if (_trail.valid())
        {
            // body forward (+X)
            osg::Vec3 fwdWorld = rot * osg::Vec3(1, 0, 0);
            osg::Vec3 tail = pos - fwdWorld * (-gTailOffset); // gTailOffset is negative
            _trail->addPoint(tail);
        }
        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
    float _tailOffset;
};

class MissileMotionCallback : public osg::NodeCallback
{
public:
    MissileMotionCallback(osg::MatrixTransform *m, Trail *trail) : mt(m), _trail(trail) {}
    osg::Vec3 pos;
    osg::Quat rot;

    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        const float dt = 0.02f;
        float t0 = std::max(0.0f, gAnim.t - dt);
        float t2 = std::min(1.0f, gAnim.t + dt);

        osg::Vec3 p0 = missileTrajectory(t0);
        osg::Vec3 p1 = missileTrajectory(gAnim.t);
        osg::Vec3 p2 = missileTrajectory(t2);

        osg::Vec3 fwd = p2 - p1;
        if (fwd.length2() < 1e-8f)
            fwd = p1 - p0;
        fwd.normalize();
        osg::Quat body = orientationFromTangent(fwd, WORLD_UP);
        rot = body * MISSILE_BASIS;
        pos = p1;

        mt->setMatrix(osg::Matrix::rotate(rot) * osg::Matrix::translate(pos));

        if (_trail.valid())
        {
            _trail->addPoint(p1 - fwd * 5.0f);
        }
        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
};

// =============== Camera utilities ===============
struct ChaseParams
{
    float back = 40.0f;   // meters behind
    float height = 10.0f; // meters above (body-up)
    float lateral = 0.0f; // meters to the right
};

// compute a chase eye/center using body vectors (NED body in world)
static void computeChaseView(const osg::Vec3 &pos, const osg::Quat &rot,
                             const ChaseParams &p, osg::Vec3 &eye, osg::Vec3 &center, osg::Vec3 &upWorld)
{
    // Body axes in world:
    const osg::Vec3 forward = rot * osg::Vec3(1, 0, 0); // +X (nose)
    const osg::Vec3 right = rot * osg::Vec3(0, 1, 0);   // +Y (right wing)
    const osg::Vec3 bodyUp = rot * osg::Vec3(0, 0, -1); // body "up" (because +Z is down)

    upWorld = WORLD_UP; // keep camera upright with world +Z

    // Chase eye a bit behind, a bit above (body-up), and optionally to the right
    eye = pos - forward * p.back + bodyUp * p.height + right * p.lateral;
    center = pos + forward * 100.0f; // look far ahead along forward
}

// ======================= Camera Updater ===========================
class ChaseCameraUpdater : public osg::NodeCallback
{
public:
    ChaseCameraUpdater(osgViewer::Viewer *v, F14MotionCallback *f, MissileMotionCallback *m)
        : viewer(v), f14CB(f), missileCB(m) {}

    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        if (!viewer)
            return;

        static int lastMode = -1;
        if (gAnim.cameraMode != lastMode)
        {
            if (gAnim.cameraMode == 0)
                viewer->setCameraManipulator(new osgGA::TrackballManipulator);
            else
                viewer->setCameraManipulator(nullptr);
            lastMode = gAnim.cameraMode;
        }

        if (gAnim.cameraMode == 0)
        {
            traverse(nullptr, nv);
            return;
        }

        osg::Vec3 eye, center, up;
        if (gAnim.cameraMode == 1 && f14CB)
        {
            ChaseParams P{45.0f, 12.0f, 0.0f};
            computeChaseView(f14CB->pos, f14CB->rot, P, eye, center, up);
        }
        else if (gAnim.cameraMode == 2 && missileCB)
        {
            ChaseParams P{22.0f, 6.0f, 0.0f};
            computeChaseView(missileCB->pos, missileCB->rot, P, eye, center, up);
        }
        else
        {
            traverse(nullptr, nv);
            return;
        }

        viewer->getCamera()->setViewMatrixAsLookAt(eye, center, up);
        traverse(nullptr, nv);
    }

private:
    osgViewer::Viewer *viewer;
    F14MotionCallback *f14CB;
    MissileMotionCallback *missileCB;
};

// =============== ImGui ===============
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *t1, Trail *t2) : _trail1(t1), _trail2(t2) {}

protected:
    void drawUi() override
    {
        ImGui::Begin("Motion Controller");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
            if (_trail1.valid())
                _trail1->clear();
            if (_trail2.valid())
                _trail2->clear();
            std::cout << ANSI_CYAN << "=== Reset motion & trails ===" << ANSI_RESET << std::endl;
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::SliderFloat("t (timeline)", &gAnim.t, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Tail Offset", &gTailOffset, -60.0f, 0.0f, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("Camera");
        const char *modes[] = {"Free", "F-14 chase", "Missile chase"};
        ImGui::Combo("Mode", &gAnim.cameraMode, modes, IM_ARRAYSIZE(modes));

        ImGui::End();
    }

private:
    osg::observer_ptr<Trail> _trail1, _trail2;
};

// =============== main ===============
int main(int, char **)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    // Ref axes
    osg::ref_ptr<osg::Node> refAxes = osgDB::readRefNodeFile(dataPath + "axes.osgt");
    osg::ref_ptr<osg::MatrixTransform> refAxesXForm = new osg::MatrixTransform;
    refAxesXForm->setMatrix(osg::Matrix::scale(5.0f, 5.0f, 5.0f));
    refAxesXForm->addChild(refAxes);
    root->addChild(refAxesXForm);

    // Trails
    osg::ref_ptr<Trail> trailF14 = new Trail(2000, 0.15f);
    osg::ref_ptr<Trail> trailMissile = new Trail(1500, 0.15f);

    // F-14
    osg::ref_ptr<osg::Node> f14 = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");
    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->setMatrix(osg::Matrix::rotate(F14_BASIS));
    aircraft->addChild(f14);
    aircraft->addChild(createAxes(15.0f));
    auto *f14CB = new F14MotionCallback(aircraft, trailF14.get(), -24.0f);
    aircraft->addUpdateCallback(f14CB);
    root->addChild(aircraft);
    root->addChild(trailF14->geode());

    // Missile
    osg::ref_ptr<osg::Node> missileModel = osgDB::readRefNodeFile(dataPath + "AIM-9L.ac");
    osg::ref_ptr<osg::MatrixTransform> missile = new osg::MatrixTransform;
    missile->addChild(missileModel);
    missile->addChild(createAxes(8.0f));
    auto *missileCB = new MissileMotionCallback(missile.get(), trailMissile.get());
    missile->addUpdateCallback(missileCB);
    root->addChild(missile);
    root->addChild(trailMissile->geode());

    // Viewer
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trailF14.get(), trailMissile.get()));

    // Start in Free mode with Trackball
    viewer.setCameraManipulator(new osgGA::TrackballManipulator);

    // Chase updater (switches manipulator automatically)
    viewer.getCamera()->setUpdateCallback(new ChaseCameraUpdater(&viewer, f14CB, missileCB));

    return viewer.run();
}
