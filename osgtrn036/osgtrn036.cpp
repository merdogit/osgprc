#include <iostream>
#include <iomanip>
#include <algorithm>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
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

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= ANSI Color Codes ===========================
#define ANSI_RESET "\e[0;0m]"
#define ANSI_BLACK "\e[0;30m"
#define ANSI_RED "\e[0;31m"
#define ANSI_GREEN "\e[0;32m"
#define ANSI_YELLOW "\e[0;33m"
#define ANSI_BLUE "\e[0;34m"
#define ANSI_MAGENTA "\e[0;35m"
#define ANSI_CYAN "\e[0;36m"
#define ANSI_WHITE "\e[0;37m"

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
    bool isFighter = true;
} gAnim;

float gTailOffset = -14.0f;
const osg::Vec3 WORLD_UP(0, 0, -1);

// ======================= Basis Adjustments ===========================
const osg::Quat F14_BASIS(-0.00622421, 0.713223, -0.700883, -0.0061165);

const osg::Quat MISSILE_BASIS( 0, 0, 1, 0);

// ======================= Trajectories ===========================
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
    // Start a bit behind and below the F-14
    float x = -120.0f + 260.0f * t + 10.0f;
    float y = 25.0f * sinf(1.2f * osg::PI * t);
    float z = -5.0f * t;
    return osg::Vec3(x, y, z);
}

// ======================= Orientation Helpers ===========================
static osg::Quat orientationFromTangent(const osg::Vec3 &forward, const osg::Vec3 &up, const bool &isFighter)
{
    osg::Vec3 X = forward;
    X.normalize();

    // In NED, body +Z is down, so invert this vector
    osg::Vec3 Z = -(up - X * (up * X));
    Z.normalize();

    // Right-hand rule: Y = Z × X
    osg::Vec3 Y = Z ^ X;
    Y.normalize();

    if (gAnim.logging)
    {
        static int frameCount = 0;
        if (frameCount % 10 == 0)
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << ANSI_CYAN << "\nBody axes in NED world:" << ANSI_RESET << "\n";
            std::cout << "  " << ANSI_RED << "+X (red, nose)  -> (" << X.x() << ", " << X.y() << ", " << X.z() << ")" << ANSI_RESET << "\n";
            std::cout << "  " << ANSI_GREEN << "+Y (green,right)-> (" << Y.x() << ", " << Y.y() << ", " << Y.z() << ")" << ANSI_RESET << "\n";
            std::cout << "  " << ANSI_BLUE << "+Z (blue,down)  -> (" << Z.x() << ", " << Z.y() << ", " << Z.z() << ")" << ANSI_RESET << "\n";
            std::cout << "----------------------------------------\n";
        }
        frameCount++;
    }

    osg::Matrix R(X.x(), Y.x(), Z.x(), 0.0f,
                  X.y(), Y.y(), Z.y(), 0.0f,
                  X.z(), Y.z(), Z.z(), 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f);

    auto rotX = R.getRotate().x();
    auto rotY = R.getRotate().y();
    auto rotZ = R.getRotate().z();
    auto rotW = R.getRotate().w();

    osg::Quat q;

    if (isFighter)
    {
        q = osg::Quat(rotX, rotZ, rotY, rotW);
    }
    else
    {
        q = osg::Quat(rotX, rotY, rotZ, rotW);
    }

    return q;
}

// ======================= Debug Axes ===========================
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

// ======================= Trail Class ===========================
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
        col->push_back(osg::Vec4(1.0f, 1.0f, 0.4f, 0.9f)); // default yellow
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
                return;
        }
        _verts->push_back(p);
        _last = p;
        _hasLast = true;

        if (_verts->size() > _maxPoints)
        {
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

        osg::Quat orient = orientationFromTangent(fwd, WORLD_UP, gAnim.isFighter);
        osg::Quat finalRot = orient * F14_BASIS;
        mt->setMatrix(osg::Matrix::rotate(finalRot) * osg::Matrix::translate(p1));

        if (_trail.valid())
        {
            osg::Vec3 worldForward = finalRot * osg::Vec3(1, 0, 0);
            osg::Vec3 tailPoint = p1 - worldForward * gTailOffset;
            _trail->addPoint(tailPoint);
        }

        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
    float _tailOffset;
};

// ======================= Missile Motion Callback ===========================
class MissileMotionCallback : public osg::NodeCallback
{
public:
    MissileMotionCallback(osg::MatrixTransform *m, Trail *trail)
        : mt(m), _trail(trail) {}

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

        osg::Quat orient = orientationFromTangent(fwd, WORLD_UP, !gAnim.isFighter);
        osg::Quat finalRot = orient * MISSILE_BASIS;
        mt->setMatrix(osg::Matrix::rotate(finalRot) * osg::Matrix::translate(p1));

        if (_trail.valid())
        {
            osg::Vec3 tail = p1 - fwd * 5.0f;
            _trail->addPoint(tail);
        }

        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
};

// ======================= ImGui Control ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *trail1, Trail *trail2)
        : _trail1(trail1), _trail2(trail2) {}

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

        ImGui::End();
    }

private:
    osg::observer_ptr<Trail> _trail1;
    osg::observer_ptr<Trail> _trail2;
};

// ======================= Main ===========================
int main(int, char **)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    osg::ref_ptr<osg::Node> refAxes = osgDB::readRefNodeFile(dataPath + "axes.osgt");
    osg::ref_ptr<osg::MatrixTransform> refAxesXForm = new osg::MatrixTransform;
    refAxesXForm->setMatrix(osg::Matrix::scale(5.0f, 5.0f, 5.0f));
    refAxesXForm->addChild(refAxes);
    root->addChild(refAxesXForm);

    // --- F14 setup ---
    osg::ref_ptr<Trail> trailF14 = new Trail(2000, 0.15f);
    root->addChild(trailF14->geode());

    osg::ref_ptr<osg::Node> f14 = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");
    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->setMatrix(osg::Matrix::rotate(F14_BASIS));
    aircraft->addChild(f14);
    aircraft->addChild(createAxes(15.0f));
    aircraft->addUpdateCallback(new F14MotionCallback(aircraft, trailF14.get(), -24.0f));
    root->addChild(aircraft);

    // --- Missile trail ---
    osg::ref_ptr<Trail> trailMissile = new Trail(1500, 0.15f);
    {
        osg::StateSet *ss = trailMissile->geode()->getOrCreateStateSet();

        osg::ref_ptr<osg::Vec4Array> col = new osg::Vec4Array;
        col->push_back(osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f)); // bright red
        osg::Geometry *geom = dynamic_cast<osg::Geometry *>(trailMissile->geode()->getDrawable(0));
        geom->setColorArray(col, osg::Array::BIND_OVERALL);

        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(2.5f);
        ss->setAttributeAndModes(lw, osg::StateAttribute::ON);

        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
    }
    root->addChild(trailMissile->geode());

    root->addChild(trailMissile->geode());

    osg::ref_ptr<osg::Node> missileModel = osgDB::readRefNodeFile(dataPath + "AIM-9L.ac");
    osg::ref_ptr<osg::MatrixTransform> missile = new osg::MatrixTransform;
    missile->addChild(missileModel);
    missile->addChild(createAxes(8.0f));
    missile->addUpdateCallback(new MissileMotionCallback(missile.get(), trailMissile.get()));
    root->addChild(missile);

    // --- Viewer ---
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trailF14.get(), trailMissile.get()));

    return viewer.run();
}