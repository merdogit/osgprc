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
        if (!gc)
            return;
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
    float t = 0.0f;
    float speed = 0.25f;
} gAnim;

float gTailOffset = -14.0f;
const osg::Vec3 WORLD_UP(0, 0, -1);

// Each model's local basis (from inspection)
const osg::Quat CESSNA_BASIS(-0.700883, 0.713223, 0.0062242, -0.00611652);
const osg::Quat TYPHOON_BASIS(0.0122325, -0.700803, 0.713142, -0.0124479);
const osg::Quat F14_BASIS(0.0, 0.713223, -0.700883, 0.0);
const osg::Quat F14_MOD_BASIS(0, 0.731354, 0, 0.681998);
const osg::Quat MISSILE_BASIS(0.0, 0.0, 1.0, 0.0);

// Pick the aircraft basis here:
const osg::Quat PLANE_BASIS = F14_MOD_BASIS;

// ===== Files for logging (trajectory + trail) =====
std::ofstream gAircraftOut;
std::ofstream gMissileOut;

// ======================= Trajectory functions ===========================
osg::Vec3 aircraftFunc(float t)
{
    float x = -120.0f + 240.0f * t;
    float y = 15.0f * sinf(1.5f * 2.0f * osg::PI * t);
    float z = 15.0f * sinf(1.5f * 2.0f * osg::PI * t);
    return osg::Vec3(x, y, z);
}

osg::Vec3 missileFunc(float t)
{
    float x = -120.0f + 260.0f * t + 10.0f;
    float y = 25.0f * sinf(1.2f * osg::PI * t);
    float z = -5.0f * sinf(3.0f * 3.0f * osg::PI * t);
    return osg::Vec3(x, y, z);
}

// ======================= File I/O ===========================
void generateTrajectoryFile(const std::string &file)
{
    std::ofstream out(file);
    if (!out)
    {
        std::cerr << "Cannot open " << file << "\n";
        return;
    }

    out << "# t ax ay az mx my mz\n";
    const int N = 500;
    for (int i = 0; i <= N; ++i)
    {
        float t = float(i) / N;
        osg::Vec3 a = aircraftFunc(t);
        osg::Vec3 m = missileFunc(t);
        out << std::fixed << std::setprecision(6)
            << t << " " << a.x() << " " << a.y() << " " << a.z() << " "
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
    if (!in)
    {
        std::cerr << "Cannot open " << file << "\n";
        return data;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        float t, ax, ay, az, mx, my, mz;
        if (ss >> t >> ax >> ay >> az >> mx >> my >> mz)
        {
            data.t.push_back(t);
            data.aircraft.push_back({ax, ay, az});
            data.missile.push_back({mx, my, mz});
        }
    }
    std::cout << "Loaded " << data.t.size() << " samples from " << file << "\n";
    return data;
}

// ======================= Metrics ===========================
struct Metrics
{
    double sumSq = 0.0;
    int n = 0;
    float lastErr = 0.0f;
    float maxErr = 0.0f;

    void add(const osg::Vec3 &expected, const osg::Vec3 &actual)
    {
        float e = (expected - actual).length();
        lastErr = e;
        maxErr = std::max(maxErr, e);
        sumSq += e * e;
        ++n;
    }

    float rmse() const { return n ? std::sqrt(sumSq / n) : 0.0f; }
    void reset() { sumSq = 0.0; n = 0; lastErr = 0.0f; maxErr = 0.0f; }
};

Metrics gAirBodyErr, gAirTrailErr;
Metrics gMisBodyErr, gMisTrailErr;

// ======================= Interpolation ===========================
osg::Vec3 interpolate(const std::vector<float> &tvec,
                      const std::vector<osg::Vec3> &vals, float t)
{
    if (tvec.empty())
        return osg::Vec3();
    if (t <= tvec.front())
        return vals.front();
    if (t >= tvec.back())
        return vals.back();
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
    if (fwd.length2() < 1e-6)
        return osg::Quat();
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
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
        ss->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);
        osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
        ss->setAttributeAndModes(lw, osg::StateAttribute::ON);

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
    }

    void add(const osg::Vec3 &p)
    {
        if (_verts->empty() || (p - _verts->back()).length() >= _minSegment)
        {
            _verts->push_back(p);
            _draw->setCount(_verts->size());
            _geom->dirtyDisplayList();
            _geom->dirtyBound();
        }
        if (_verts->size() > _maxPoints)
        {
            const size_t overflow = _verts->size() - _maxPoints;
            _verts->erase(_verts->begin(), _verts->begin() + overflow);
            _draw->setCount(_verts->size());
        }
    }

    bool lastPoint(osg::Vec3 &out) const
    {
        if (_verts->empty())
            return false;
        out = _verts->back();
        return true;
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
            gAnim.t += gAnim.speed * 0.01f;
            if (gAnim.t > 1.0f) gAnim.t = 1.0f;
        }

        osg::Vec3 pPrev = interpolate(data->t, data->aircraft, std::max(0.0f, gAnim.t - 0.01f));
        osg::Vec3 p     = interpolate(data->t, data->aircraft, gAnim.t);
        osg::Vec3 pNext = interpolate(data->t, data->aircraft, std::min(1.0f, gAnim.t + 0.01f));

        osg::Quat q = PLANE_BASIS * orientationFromTrajectory(pPrev, p, pNext);
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));

        osg::Vec3 matPos = mt->getMatrix().getTrans();
        gAirBodyErr.add(p, matPos);

        if (trail.valid())
        {
            osg::Vec3 bodyForward = q * (PLANE_BASIS * osg::Vec3(1, 0, 0));
            osg::Vec3 tail = p - bodyForward * gTailOffset;
            trail->add(tail);

            osg::Vec3 last;
            if (trail->lastPoint(last))
                gAirTrailErr.add(p, last);

            // ---- Log to file: t, trajectory (p), trail (tail)
            if (gAnim.running && gAnim.t > 0.0f && gAircraftOut.is_open())
            {
                gAircraftOut << std::fixed << std::setprecision(6)
                             << gAnim.t << " "
                             << p.x()    << " " << p.y()    << " " << p.z()    << " "
                             << tail.x() << " " << tail.y() << " " << tail.z() << "\n";
            }
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
        osg::Vec3 p  = interpolate(data->t, data->missile, gAnim.t);
        osg::Vec3 p2 = interpolate(data->t, data->missile, std::min(1.0f, gAnim.t + 0.01f));
        osg::Quat q  = MISSILE_BASIS * orientationFromTrajectory(p, p, p2);

        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));

        osg::Vec3 matPos = mt->getMatrix().getTrans();
        gMisBodyErr.add(p, matPos);

        if (trail.valid())
        {
            osg::Vec3 bodyForward = q * (MISSILE_BASIS * osg::Vec3(1, 0, 0));
            osg::Vec3 tail = p - bodyForward * 5.0f;
            trail->add(tail);

            osg::Vec3 last;
            if (trail->lastPoint(last))
                gMisTrailErr.add(p, last);

            // ---- Log to file: t, trajectory (p), trail (tail)
            if (gAnim.running && gAnim.t > 0.0f && gMissileOut.is_open())
            {
                gMissileOut << std::fixed << std::setprecision(6)
                            << gAnim.t << " "
                            << p.x()    << " " << p.y()    << " " << p.z()    << " "
                            << tail.x() << " " << tail.y() << " " << tail.z() << "\n";
            }
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
        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f;
            gAnim.running = false;
            if (ta.valid()) ta->clear();
            if (tm.valid()) tm->clear();
            gAirBodyErr.reset();
            gAirTrailErr.reset();
            gMisBodyErr.reset();
            gMisTrailErr.reset();

            // Mark new segment in logs (optional)
            if (gAircraftOut.is_open()) gAircraftOut << "# --- reset ---\n";
            if (gMissileOut.is_open())  gMissileOut  << "# --- reset ---\n";

            std::cout << "=== Trails cleared and animation reset ===\n";
        }
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::SliderFloat("t", &gAnim.t, 0.0f, 1.0f, "%.3f");

        ImGui::Separator();
        ImGui::Text("Error metrics (traj vs actual)");
        ImGui::Text("Aircraft body : last %.4f  max %.4f  RMSE %.4f",
                    gAirBodyErr.lastErr, gAirBodyErr.maxErr, gAirBodyErr.rmse());
        ImGui::Text("Aircraft trail: last %.4f  max %.4f  RMSE %.4f",
                    gAirTrailErr.lastErr, gAirTrailErr.maxErr, gAirTrailErr.rmse());
        ImGui::Text("Missile body  : last %.4f  max %.4f  RMSE %.4f",
                    gMisBodyErr.lastErr, gMisBodyErr.maxErr, gMisBodyErr.rmse());
        ImGui::Text("Missile trail : last %.4f  max %.4f  RMSE %.4f",
                    gMisTrailErr.lastErr, gMisTrailErr.maxErr, gMisTrailErr.rmse());

        ImGui::End();
    }

    osg::observer_ptr<Trail> ta, tm;
};

// ======================= Main ===========================
int main()
{
    const std::string baseDir  = "/home/murate/Documents/SwTrn/OsgPrc/osgtrn059/";
    const std::string trajFile = baseDir + "trajectory.txt";
    generateTrajectoryFile(trajFile);
    TrajData data = loadTrajectoryFile(trajFile);

    // Open log files
    gAircraftOut.open(baseDir + "aircraft_traj_trail.txt");
    gMissileOut.open(baseDir + "missile_traj_trail.txt");
    if (gAircraftOut.is_open())
        gAircraftOut << "# t ax ay az trailx traily trailz\n";
    if (gMissileOut.is_open())
        gMissileOut << "# t mx my mz trailx traily trailz\n";

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    osg::ref_ptr<Trail> trailAircraft = new Trail(2000, 0.15f);
    osg::ref_ptr<Trail> trailMissile  = new Trail(1500, 0.15f);

    // Missile trail red
    {
        osg::StateSet *ss = trailMissile->geode()->getOrCreateStateSet();
        osg::ref_ptr<osg::Vec4Array> col = new osg::Vec4Array;
        col->push_back(osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f));
        osg::Geometry *geom = dynamic_cast<osg::Geometry *>(trailMissile->geode()->getDrawable(0));
        geom->setColorArray(col, osg::Array::BIND_OVERALL);
    }

    root->addChild(trailAircraft->geode());
    root->addChild(trailMissile->geode());

    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgPrc/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> acraft  = osgDB::readRefNodeFile(dataPath + "F-14-low-poly-axes-modified.ac");
    osg::ref_ptr<osg::Node> missile = osgDB::readRefNodeFile(dataPath + "AIM-9L.ac");

    osg::ref_ptr<osg::MatrixTransform> air = new osg::MatrixTransform;
    air->addChild(acraft);
    osg::ref_ptr<osg::MatrixTransform> mis = new osg::MatrixTransform;
    mis->addChild(missile);
    air->addUpdateCallback(new AircraftCB(air.get(), trailAircraft.get(), &data));
    mis->addUpdateCallback(new MissileCB(mis.get(), trailMissile.get(), &data));
    root->addChild(air);
    root->addChild(mis);

    osg::ref_ptr<osg::Node> refAxes = osgDB::readNodeFile(dataPath + "axes.osgt");
    osg::ref_ptr<osg::MatrixTransform> refAxesXForm = new osg::MatrixTransform;
    refAxesXForm->addChild(refAxes);
    refAxesXForm->setMatrix(osg::Matrix::scale(2.0f, 2.0f, 2.0f));
    root->addChild(refAxesXForm.get());

    osg::ref_ptr<osgGA::NodeTrackerManipulator> man = new osgGA::NodeTrackerManipulator;
    man->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
    man->setTrackNode(missile);
    man->setHomePosition(osg::Vec3d(-100, 0, -25), osg::Vec3d(0, 0, 0), osg::Vec3d(0, 0, -1));

    osgViewer::Viewer viewer;
    viewer.setCameraManipulator(man);
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl(trailAircraft.get(), trailMissile.get()));

    int rc = viewer.run();

    if (gAircraftOut.is_open()) gAircraftOut.close();
    if (gMissileOut.is_open())  gMissileOut.close();

    return rc;
}
