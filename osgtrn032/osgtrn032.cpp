#include <iostream>
#include <algorithm>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osgDB/ReadFile>
#include <osg/Math>
#include <osg/Material>
#include <osg/BlendFunc>

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
        auto *gc = dynamic_cast<osg::GraphicsContext *>(object);
        if (!gc)
            return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ======================= Global Animation State ===========================
struct AnimationState
{
    bool running = false;
    bool logging = false;
    float t = 0.0f;
    float speed = 0.25f;
} gAnim;

const osg::Vec3 WORLD_UP(0, 0, 1);

// ======================= Trajectory ===========================
static inline float easeCos01(float t)
{
    return 0.5f * (1.0f - cosf(osg::PI * std::clamp(t, 0.0f, 1.0f)));
}
osg::Vec3 aircraftTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float x = -60.0f + 120.0f * t;        // +X forward
    float y = 20.0f * easeCos01(t);       // lateral arc
    float z = 5.0f + 8.0f * easeCos01(t); // gentle climb
    return osg::Vec3(x, y, z);
}

// ======================= Orientation Helpers ===========================
static osg::Quat orientationFromTangent(const osg::Vec3 &forward, const osg::Vec3 &up)
{
    osg::Vec3 X = forward;
    X.normalize();
    osg::Vec3 Z = up - X * (up * X);
    Z.normalize(); // wings level wrt world up
    osg::Vec3 Y = Z ^ X;
    Y.normalize();

    osg::Matrix R(X.x(), Y.x(), Z.x(), 0.0f,
                  X.y(), Y.y(), Z.y(), 0.0f,
                  X.z(), Y.z(), Z.z(), 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f);
    osg::Quat q;
    q.set(R);
    return q;
}

// Model basis fix
static osg::Quat makeF14Basis()
{
    osg::Matrix M(-1, 0, 0, 0,
                  0, 0, 1, 0,
                  0, 1, 0, 0,
                  0, 0, 0, 1);
    osg::Quat q;
    q.set(M);
    return q;
}
// const osg::Quat F14_BASIS = makeF14Basis();
const osg::Quat F14_BASIS =
    osg::Quat(osg::PI, osg::Vec3(0, 0, 1)) *
    osg::Quat(osg::PI_2, osg::Vec3(1, 0, 0));
// const osg::Quat F14_BASIS(-0.00622421, 0.713223, -0.700883, -0.0061165);

// const osg::Quat F14_BASIS = osg::Quat(-0.00622421, 0.713223, -0.700883, -0.0061165);

// ======================= Debug Axes (X=red, Y=green, Z=blue) ===========================
osg::ref_ptr<osg::Geode> createAxes(float len = 10.0f)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array();

    // X (red) – reversed
    v->push_back({0, 0, 0});
    v->push_back({-len, 0, 0});
    c->push_back({1, 0, 0, 1});
    c->push_back({1, 0, 0, 1});

    // Y (green) – swapped Z↔Y and reversed
    v->push_back({0, 0, 0});
    v->push_back({0, 0, -len});
    c->push_back({0, 1, 0, 1});
    c->push_back({0, 1, 0, 1});

    // Z (blue) – swapped Y↔Z and reversed
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

// ======================= Simple Trail (world-space LINE_STRIP) ===========================
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

        // color: light yellow with slight alpha
        osg::ref_ptr<osg::Vec4Array> col = new osg::Vec4Array;
        col->push_back(osg::Vec4(1.0f, 1.0f, 0.4f, 0.9f));
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
        if (_hasLast)
        {
            if ((p - _last).length() < _minSegment)
                return; // skip tiny segments
        }
        _verts->push_back(p);
        _last = p;
        _hasLast = true;

        // clamp size
        if (_verts->size() > _maxPoints)
        {
            // pop from front — cheap way: shift by erasing first N
            const size_t overflow = _verts->size() - _maxPoints;
            _verts->erase(_verts->begin(), _verts->begin() + overflow);
        }

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

// ======================= F-14 Motion Callback ===========================
class F14MotionCallback : public osg::NodeCallback
{
public:
    F14MotionCallback(osg::MatrixTransform *m, Trail *trail, float tailOffset = 4.0f)
        : mt(m), _trail(trail), _tailOffset(tailOffset) {}

    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        if (gAnim.running)
        {
            if (!gAnim.logging)
            {
                std::cout << "\n=== Logging started ===" << std::endl;
                gAnim.logging = true;
            }
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t >= 1.0f)
            {
                gAnim.t = 1.0f;
                gAnim.running = false;
                if (gAnim.logging)
                {
                    std::cout << "=== Logging stopped ===\n"
                              << std::endl;
                    gAnim.logging = false;
                }
            }
        }

        // sample trajectory & tangent
        const float dt = 0.02f;
        float t0 = std::max(0.0f, gAnim.t - dt);
        float t2 = std::min(1.0f, gAnim.t + dt);

        osg::Vec3 p0 = aircraftTrajectory(t0);
        osg::Vec3 p1 = aircraftTrajectory(gAnim.t);
        osg::Vec3 p2 = aircraftTrajectory(t2);

        osg::Vec3 fwd = p2 - p1;
        fwd.normalize();
        osg::Quat orient = orientationFromTangent(fwd, WORLD_UP);
        osg::Quat finalRot = orient * F14_BASIS;

        // place model
        mt->setMatrix(osg::Matrix::rotate(finalRot) * osg::Matrix::translate(p1));

        // --- add trail point at the "rear" of the aircraft (behind along -forward) ---
        if (_trail.valid())
        {
            osg::Vec3 worldForward = finalRot * osg::Vec3(1, 0, 0); // +X is forward
            osg::Vec3 tailPoint = p1 - worldForward * _tailOffset;
            _trail->addPoint(tailPoint);
        }

        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
    float _tailOffset; // distance from model origin to tail sample point
};

// ======================= ImGui ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *trail) : _trail(trail) {}

protected:
    void drawUi() override
    {
        ImGui::Begin("F-14 Motion");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
        {
            gAnim.running = !gAnim.running;
            if (gAnim.running)
            {
                gAnim.logging = true;
                std::cout << "\n=== Logging started ===" << std::endl;
            }
            else if (gAnim.logging)
            {
                std::cout << "=== Logging stopped ===\n"
                          << std::endl;
                gAnim.logging = false;
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
            if (gAnim.logging)
            {
                std::cout << "=== Logging stopped ===\n"
                          << std::endl;
                gAnim.logging = false;
            }
            if (_trail.valid())
                _trail->clear(); // <<< clear trail
            std::cout << "=== Reset motion & trail ===" << std::endl;
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::End();
    }

private:
    osg::observer_ptr<Trail> _trail;
};

// ======================= Main ===========================
int main(int, char **)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    // osg::ref_ptr<osg::Node> refAxes = osgDB::readRefNodeFile(dataPath + "axes.osgt");
    // root->addChild(refAxes);

    // --- Trail node (world-space, separate from the aircraft transform) ---
    osg::ref_ptr<Trail> trail = new Trail(/*maxPoints*/ 2000, /*minSegment*/ 0.15f);
    root->addChild(trail->geode());

    // --- Model ---
    osg::ref_ptr<osg::Node> f14 = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");

    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->addChild(f14);
    aircraft->addChild(createAxes(15.0f));
    aircraft->addUpdateCallback(new F14MotionCallback(aircraft, trail.get(), /*tailOffset*/ 4.0f));
    root->addChild(aircraft);

    // --- Viewer ---
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trail.get()));

    return viewer.run();
}