#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgGA/NodeTrackerManipulator>
#include <osg/LineWidth>
#include <osg/BlendFunc>
#include <osg/Geometry>
#include <osg/Geode>
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
        if (!gc) return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
        else
            std::cout << "ImGui OpenGL3 initialized.\n";
    }
};

// ======================= Globals ===========================
struct AnimationState
{
    bool running = false;
    float simTime = 0.0f;      // seconds
    float totalTime = 10.0f;   // total duration (seconds)
    float speed = 1.0f;        // time multiplier
    bool interpolate = true;   // <--- toggle interpolation
} gAnim;

const float DT_STEP = 0.1f; // 100 ms step
float gTailOffset = -14.0f;
const osg::Vec3 WORLD_UP(0, 0, -1);

// ======================= Bases ===========================
const osg::Quat F14_MOD_BASIS(0, 0.731354, 0, 0.681998);
const osg::Quat MISSILE_BASIS(0.0, 0.0, 1.0, 0.0);
const osg::Quat PLANE_BASIS = F14_MOD_BASIS;

// ===== Files for logging =====
std::ofstream gAircraftOut;
std::ofstream gMissileOut;

// ======================= Trajectory functions ===========================
osg::Vec3 aircraftFunc(float t)
{
    float x = -120.0f + 24.0f * t;
    float y = 15.0f * sinf(1.5f * osg::PI * t / gAnim.totalTime * 2.0f);
    float z = 15.0f * sinf(1.5f * osg::PI * t / gAnim.totalTime * 2.0f);
    return osg::Vec3(x, y, z);
}

osg::Vec3 missileFunc(float t)
{
    float x = -110.0f + 26.0f * t;
    float y = 25.0f * sinf(1.2f * osg::PI * t / gAnim.totalTime);
    float z = -5.0f * sinf(3.0f * osg::PI * t / gAnim.totalTime);
    return osg::Vec3(x, y, z);
}

// ======================= File I/O ===========================
void generateTrajectoryFile(const std::string &file)
{
    std::ofstream out(file);
    out << "# time(s) ax ay az mx my mz\n";
    int N = static_cast<int>(gAnim.totalTime / DT_STEP);
    for (int i = 0; i <= N; ++i)
    {
        float t = i * DT_STEP;
        osg::Vec3 a = aircraftFunc(t);
        osg::Vec3 m = missileFunc(t);
        out << std::fixed << std::setprecision(3)
            << t << " "
            << a.x() << " " << a.y() << " " << a.z() << " "
            << m.x() << " " << m.y() << " " << m.z() << "\n";
    }
    std::cout << "Trajectory file written: " << file << "\n";
}

struct TrajData
{
    std::vector<float> t;
    std::vector<osg::Vec3> aircraft, missile;
};

TrajData loadTrajectoryFile(const std::string &file)
{
    TrajData data;
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        float t, ax, ay, az, mx, my, mz;
        if (ss >> t >> ax >> ay >> az >> mx >> my >> mz)
        {
            data.t.push_back(t);
            data.aircraft.push_back({ax, ay, az});
            data.missile.push_back({mx, my, mz});
        }
    }
    gAnim.totalTime = data.t.back();
    return data;
}

// ======================= Interpolation ===========================
osg::Vec3 interpolate(const std::vector<float> &tvec,
                      const std::vector<osg::Vec3> &vals, float t)
{
    if (!gAnim.interpolate) {
        // Non-interpolated mode: snap to nearest sample
        size_t idx = 0;
        float minDiff = 1e9;
        for (size_t i = 0; i < tvec.size(); ++i) {
            float diff = std::abs(t - tvec[i]);
            if (diff < minDiff) { minDiff = diff; idx = i; }
        }
        return vals[idx];
    }

    // Interpolated mode (same as before)
    if (tvec.empty()) return osg::Vec3();
    if (t <= tvec.front()) return vals.front();
    if (t >= tvec.back()) return vals.back();
    for (size_t i = 1; i < tvec.size(); ++i)
    {
        if (t < tvec[i])
        {
            float u = (t - tvec[i - 1]) / (tvec[i] - tvec[i - 1]);
            return vals[i - 1] * (1.0f - u) + vals[i] * u;
        }
    }
    return vals.back();
}

// ======================= Orientation with banking ===========================
static osg::Quat orientationFromTrajectory(const osg::Vec3 &pPrev,
                                           const osg::Vec3 &p,
                                           const osg::Vec3 &pNext)
{
    osg::Vec3 fwd = pNext - p;
    if (fwd.length2() < 1e-6) return osg::Quat();
    fwd.normalize();

    float yaw = atan2f(fwd.y(), fwd.x());
    float pitch = asinf(-fwd.z());

    osg::Vec3 prevDir = p - pPrev;
    prevDir.normalize();
    osg::Vec3 turnAxis = prevDir ^ fwd;
    float turnStrength = turnAxis.length();
    float bankSign = (turnAxis.z() > 0.0f) ? -1.0f : 1.0f;
    float roll = bankSign * turnStrength * 1.8f;

    osg::Quat qYaw(yaw, osg::Vec3(0, 0, 1));
    osg::Quat qPitch(pitch, osg::Vec3(0, 1, 0));
    osg::Quat qRoll(roll, osg::Vec3(1, 0, 0));
    return qYaw * qPitch * qRoll;
}

// ======================= Trail ===========================
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
        col->push_back(osg::Vec4(1.0f, 1.0f, 0.2f, 1.0f));
        _geom->setColorArray(col, osg::Array::BIND_OVERALL);
        osg::StateSet *ss = _geom->getOrCreateStateSet();
        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
        ss->setAttributeAndModes(lw, osg::StateAttribute::ON);
        _geode = new osg::Geode;
        _geode->addDrawable(_geom.get());
    }
    osg::Geode *geode() const { return _geode.get(); }
    void clear() { _verts->clear(); _draw->setCount(0); _geom->dirtyDisplayList(); _geom->dirtyBound(); }
    void add(const osg::Vec3 &p)
    {
        if (_verts->empty() || (p - _verts->back()).length() >= _minSegment)
        {
            _verts->push_back(p);
            _draw->setCount(_verts->size());
            _geom->dirtyDisplayList();
            _geom->dirtyBound();
        }
    }
private:
    osg::ref_ptr<osg::Vec3Array> _verts;
    osg::ref_ptr<osg::Geometry> _geom;
    osg::ref_ptr<osg::DrawArrays> _draw;
    osg::ref_ptr<osg::Geode> _geode;
    size_t _maxPoints;
    float _minSegment;
};

// ======================= Motion Callbacks ===========================
class AircraftCB : public osg::NodeCallback
{
public:
    AircraftCB(osg::MatrixTransform *m, Trail *t, const TrajData *d)
        : mt(m), trail(t), data(d) {}
    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        if (gAnim.running)
        {
            gAnim.simTime += DT_STEP * gAnim.speed;
            if (gAnim.simTime > gAnim.totalTime) gAnim.simTime = gAnim.totalTime;
        }
        float t = gAnim.simTime;
        osg::Vec3 pPrev = interpolate(data->t, data->aircraft, std::max(0.0f, t - DT_STEP));
        osg::Vec3 p     = interpolate(data->t, data->aircraft, t);
        osg::Vec3 pNext = interpolate(data->t, data->aircraft, std::min(gAnim.totalTime, t + DT_STEP));
        osg::Quat q = PLANE_BASIS * orientationFromTrajectory(pPrev, p, pNext);
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));
        if (trail.valid() && gAnim.running)
        {
            osg::Vec3 bodyForward = q * (PLANE_BASIS * osg::Vec3(1, 0, 0));
            trail->add(p - bodyForward * gTailOffset);
        }
        traverse(mt.get(), nv);
    }
private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> trail;
    const TrajData *data;
};

class MissileCB : public osg::NodeCallback
{
public:
    MissileCB(osg::MatrixTransform *m, Trail *t, const TrajData *d)
        : mt(m), trail(t), data(d) {}
    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        float t = gAnim.simTime;
        osg::Vec3 p  = interpolate(data->t, data->missile, t);
        osg::Vec3 p2 = interpolate(data->t, data->missile, std::min(gAnim.totalTime, t + DT_STEP));
        osg::Quat q  = MISSILE_BASIS * orientationFromTrajectory(p, p, p2);
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));
        if (trail.valid() && gAnim.running)
        {
            osg::Vec3 bodyForward = q * (MISSILE_BASIS * osg::Vec3(1, 0, 0));
            trail->add(p - bodyForward * 5.0f);
        }
        traverse(mt.get(), nv);
    }
private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail> trail;
    const TrajData *data;
};

// ======================= ImGui UI ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *a, Trail *m) : ta(a), tm(m) {}
protected:
    void drawUi() override
    {
        ImGui::Begin("Trajectory Control");
        if (ImGui::Button(gAnim.running ? "Stop" : "Start")) gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset")) { gAnim.simTime = 0.0f; gAnim.running = false; ta->clear(); tm->clear(); }
        ImGui::Checkbox("Interpolation", &gAnim.interpolate);
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.1f, 2.0f, "%.2f");
        ImGui::SliderFloat("Time (s)", &gAnim.simTime, 0.0f, gAnim.totalTime, "%.2f");
        ImGui::End();
    }
    osg::observer_ptr<Trail> ta, tm;
};

// ======================= Main ===========================
int main()
{
    const std::string baseDir  = "/home/murate/Documents/SwTrn/OsgTrn/osgtrn061/";
    const std::string trajFile = baseDir + "trajectory.txt";
    generateTrajectoryFile(trajFile);
    TrajData data = loadTrajectoryFile(trajFile);

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<Trail> trailAircraft = new Trail(2000, 0.15f);
    osg::ref_ptr<Trail> trailMissile  = new Trail(1500, 0.15f);
    root->addChild(trailAircraft->geode());
    root->addChild(trailMissile->geode());

    osg::ref_ptr<osg::Node> acraft  = osgDB::readRefNodeFile("/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/F-14-low-poly-axes-modified.ac");
    osg::ref_ptr<osg::Node> missile = osgDB::readRefNodeFile("/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/AIM-9L.ac");

    osg::ref_ptr<osg::MatrixTransform> air = new osg::MatrixTransform;
    air->addChild(acraft);
    osg::ref_ptr<osg::MatrixTransform> mis = new osg::MatrixTransform;
    mis->addChild(missile);
    air->addUpdateCallback(new AircraftCB(air.get(), trailAircraft.get(), &data));
    mis->addUpdateCallback(new MissileCB(mis.get(), trailMissile.get(), &data));
    root->addChild(air);
    root->addChild(mis);

    osgViewer::Viewer viewer;
    osg::ref_ptr<osgGA::NodeTrackerManipulator> man = new osgGA::NodeTrackerManipulator;
    man->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
    man->setTrackNode(acraft);
    viewer.setCameraManipulator(man);
    viewer.apply(new osgViewer::SingleWindow(100,100,1000,700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trailAircraft.get(), trailMissile.get()));
    return viewer.run();
}