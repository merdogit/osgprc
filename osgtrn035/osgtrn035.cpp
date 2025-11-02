#include <iostream>
#include <iomanip>
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

// ======================= ANSI Color Codes ===========================
// #define ANSI_RESET "\033[0m"
// #define ANSI_RED "\033[31m"
// #define ANSI_GREEN "\033[32m"
// #define ANSI_BLUE "\033[34m"
// #define ANSI_YELLOW "\033[33m"
// #define ANSI_CYAN "\033[36m"

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
} gAnim;

float gTailOffset = 24.0f;

const osg::Vec3 WORLD_UP(0, 0, -1);

// ======================= F-14 Basis Adjustment ===========================
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

static osg::Quat makeF14Basis2()
{
    osg::Matrix M(1, 0, 0, 0,
                  0, 0, -1, 0,
                  0, 1, 0, 0,
                  0, 0, 0, 1);

    // osg::Quat r(-0.00622421, 0.713223, -0.700883, -0.0061165);
    osg::Quat q;
    q.set(M);

    return q;
}

static osg::Matrix makeNedToOsgMatrix()
{
    // | 1  0  0  0 |
    // | 0  0 -1  0 |
    // | 0  1  0  0 |
    // | 0  0  0  1 |
    return osg::Matrix(1, 0, 0, 0,
                       0, 0, -1, 0,
                       0, 1, 0, 0,
                       0, 0, 0, 1);
}

static osg::Quat makeOrientationNedToOsgMatrix()
{
    // | 1  0  0  0 |
    // | 0  0 -1  0 |
    // | 0  1  0  0 |
    // | 0  0  0  1 |
    osg::Matrix M(1, 0, 0, 0,
                  0, 0, -1, 0,
                  0, 1, 0, 0,
                  0, 0, 0, 1);

    osg::Quat q;
    q.set(M);

    return q;
}

// const osg::Quat F14_BASIS =
//     osg::Quat(osg::PI, osg::Vec3(0, 0, 1)) *
//     osg::Quat(osg::PI_2, osg::Vec3(1, 0, 0));

const osg::Quat F14_BASIS(-0.00622421, 0.713223, -0.700883, -0.0061165);

// ======================= Trajectory ===========================
static inline float easeCos01(float t)
{
    return 0.5f * (1.0f - cosf(osg::PI * std::clamp(t, 0.0f, 1.0f)));
}

// ======================= Trajectory (Step 1: Z-axis oscillation only) ===========================
osg::Vec3 aircraftTrajectoryZOscillation(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);

    // Forward motion along X (straight line)
    float x = -120.0f + 240.0f * t;

    // No lateral deviation yet
    float y = 0.0f;

    // Simple vertical oscillation in ± range
    const float amplitude = 15.0f; // how high/low (± range)
    const float cycles = 1.5f;     // number of up-down waves along the path
    float z = amplitude * sinf(cycles * 2.0f * osg::PI * t);

    if (gAnim.logging)
    {
        static int frameCount = 0;
        if (frameCount % 10 == 0)
        {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "----------------------------------------\n";
            std::cout << "t = " << t
                      << "   x = " << x
                      << "   y = " << y
                      << "   z = " << z << '\n';
        }
        frameCount++;
    }

    return osg::Vec3(x, y, z);
}

osg::Vec3 aircraftTrajectoryYOscillation(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);

    // Forward motion along X (straight line)
    float x = -120.0f + 240.0f * t;

    // Lateral oscillation in Y (left-right wave)
    const float amplitude = 15.0f; // ± range of side movement
    const float cycles = 1.5f;     // number of side-to-side waves along the path
    float y = amplitude * sinf(cycles * 2.0f * osg::PI * t);

    // Keep Z constant (level flight)
    float z = 0.0f;

    if (gAnim.logging)
    {
        static int frameCount = 0;
        if (frameCount % 10 == 0)
        {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "----------------------------------------\n";
            std::cout << "t = " << t
                      << "   x = " << x
                      << "   y = " << y
                      << "   z = " << z << '\n';
        }
        frameCount++;
    }

    return osg::Vec3(x, y, z);
}

osg::Vec3 aircraftTrajectory(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);

    // Forward motion along X (straight line)
    float x = -120.0f + 240.0f * t;

    // Lateral oscillation in Y (left-right wave)
    const float amplitude = 15.0f; // ± range of side movement
    const float cycles = 1.5f;     // number of side-to-side waves along the path

    float y = amplitude * sinf(cycles * 2.0f * osg::PI * t);
    float z = amplitude * sinf(cycles * 2.0f * osg::PI * t);

    if (gAnim.logging)
    {
        static int frameCount = 0;
        if (frameCount % 10 == 0)
        {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "----------------------------------------\n";
            std::cout << "t = " << t
                      << "   x = " << x
                      << "   y = " << y
                      << "   z = " << z << '\n';
        }
        frameCount++;
    }

    return osg::Vec3(x, y, z);
}

// ======================= Orientation Helpers ===========================
static osg::Quat orientationFromTangent(const osg::Vec3 &forward, const osg::Vec3 &up)
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

    osg::Quat q(rotX, rotZ, rotY, rotW);

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
        // Advance time only when running
        if (gAnim.running)
        {
            if (!gAnim.logging)
            {
                std::cout << ANSI_YELLOW << "\n=== Logging started ===" << ANSI_RESET << std::endl;
                gAnim.logging = true;
            }
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t >= 1.0f)
            {
                gAnim.t = 1.0f;
                gAnim.running = false;
                if (gAnim.logging)
                {
                    std::cout << ANSI_YELLOW << "=== Logging stopped ===" << ANSI_RESET << "\n"
                              << std::endl;
                    gAnim.logging = false;
                }
            }
        }

        // Always compute pose for current gAnim.t so slider scrubbing updates the model
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

        osg::Quat orient = orientationFromTangent(fwd, WORLD_UP);
        osg::Quat finalRot = orient * F14_BASIS;

        mt->setMatrix(osg::Matrix::rotate(finalRot) * osg::Matrix::translate(p1));

        // Only leave a trail when running (scrubbing won't paint lines)
        if (_trail.valid())
        {
            osg::Vec3 worldForward = finalRot * osg::Vec3(1, 0, 0);
            osg::Vec3 tailPoint = p1 - worldForward * gTailOffset; // gTailOffset should be POSITIVE distance; sign handled here
            _trail->addPoint(tailPoint);
        }

        traverse(mt.get(), nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> _trail;
    float _tailOffset;
};

// ======================= ImGui Control ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *trail) : _trail(trail) {}

protected:
    void drawUi() override
    {
        ImGui::Begin("F-14 Motion Controller");

        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
        {
            gAnim.running = !gAnim.running;
            if (gAnim.running)
            {
                gAnim.logging = true;
                std::cout << ANSI_YELLOW << "\n=== Logging started ===" << ANSI_RESET << std::endl;
            }
            else if (gAnim.logging)
            {
                std::cout << ANSI_YELLOW << "=== Logging stopped ===" << ANSI_RESET << "\n"
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
                std::cout << ANSI_YELLOW << "=== Logging stopped ===" << ANSI_RESET << "\n"
                          << std::endl;
                gAnim.logging = false;
            }
            if (_trail.valid())
                _trail->clear();
            std::cout << ANSI_CYAN << "=== Reset motion & trail ===" << ANSI_RESET << std::endl;
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");

        // ---- Timeline (compatible header) ----
        ImGui::TextUnformatted("Simulation Progress");
        ImGui::Separator();

        static bool wasRunning = false;
        if (ImGui::SliderFloat("t (timeline)", &gAnim.t, 0.0f, 1.0f, "%.3f"))
        {
            // While scrubbing, pause auto-advance
            if (ImGui::IsItemActivated())
            {
                wasRunning = gAnim.running;
                gAnim.running = false;
            }
            // Clear trail after scrub to avoid discontinuity
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                if (_trail.valid())
                    _trail->clear();
                gAnim.running = wasRunning;
            }
        }

        ImGui::SliderFloat("Tail Offset", &gTailOffset, -60.0f, 0.0f, "%.1f");

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

    osg::ref_ptr<osg::Node> refAxes = osgDB::readRefNodeFile(dataPath + "axes.osgt");
    osg::ref_ptr<osg::MatrixTransform> refAxesXForm = new osg::MatrixTransform;
    refAxesXForm->setMatrix(osg::Matrix::scale(5.0f, 5.0f, 5.0f));
    refAxesXForm->addChild(refAxes);
    root->addChild(refAxesXForm);

    osg::ref_ptr<Trail> trail = new Trail(2000, 0.15f);
    root->addChild(trail->geode());

    osg::ref_ptr<osg::Node> f14 = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");

    osg::ref_ptr<osg::MatrixTransform> aircraft = new osg::MatrixTransform;
    aircraft->setMatrix(osg::Matrix::rotate(F14_BASIS));
    aircraft->addChild(f14);
    aircraft->addChild(createAxes(15.0f));
    aircraft->addUpdateCallback(new F14MotionCallback(aircraft, trail.get(), -24.0f));
    root->addChild(aircraft);

    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trail.get()));

    return viewer.run();
}