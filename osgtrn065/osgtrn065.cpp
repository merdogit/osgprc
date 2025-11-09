// osgtrn065.cpp
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/LineWidth>
#include <osg/BlendFunc>
#include <osgDB/ReadFile>
#include <osgGA/OrbitManipulator>
#include <osgViewer/CompositeViewer>
#include <osgViewer/View>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/GraphicsWindow>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

// Your existing ImGui glue (assumed available)
#include "OsgImGuiHandler.hpp"

// ======================= ImGui GC Init ===========================
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
    bool  running   = false;
    float simTime   = 0.0f;     // seconds
    float totalTime = 10.0f;    // seconds
    float speed     = 1.0f;     // time multiplier
} gAnim;

const float DT_STEP = 0.1f; // 100 ms fixed time step
float gTailOffset   = -14.0f;
const osg::Vec3 WORLD_UP(0,0,-1);

// ======================= Modes & missile params ===========================
struct MissileParams {
    bool  dynamicChase   = false; // OFF = use file; ON = steer to fighter
    float speedMS        = 30.0f; // m/s (scene scale numbers)
    float turnRateDegPS  = 60.0f; // deg/s (max yaw/pitch rate)
} gMissileParam;

struct MissileState {
    osg::Vec3 p = osg::Vec3(-110.0f, 0.0f, 0.0f);
    osg::Vec3 v = osg::Vec3(  26.0f, 0.0f, 0.0f); // initial forward-ish
    void reset()
    {
        p.set(-110.0f, 0.0f, 0.0f);
        // orient roughly along +X with chosen speed
        v = osg::Vec3(1,0,0) * gMissileParam.speedMS;
    }
} gMissile;

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
    float x = -120.0f + 24.0f * t; // 24 m/s forward
    float y =  15.0f * sinf(1.5f * osg::PI * t / gAnim.totalTime * 2.0f);
    float z =  15.0f * sinf(1.5f * osg::PI * t / gAnim.totalTime * 2.0f);
    return osg::Vec3(x, y, z);
}
osg::Vec3 missileFunc(float t)
{
    // (Used only for file mode / visualization parity)
    float x = -110.0f + 26.0f * t;
    float y =  25.0f * sinf(1.2f * osg::PI * t / gAnim.totalTime);
    float z =  -5.0f * sinf(3.0f * osg::PI * t / gAnim.totalTime);
    return osg::Vec3(x, y, z);
}

// ======================= File I/O ===========================
void generateTrajectoryFile(const std::string &file)
{
    std::ofstream out(file);
    if (!out) { std::cerr << "Cannot open " << file << "\n"; return; }

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
    if (!in) { std::cerr << "Cannot open " << file << "\n"; return data; }

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
    if (!data.t.empty()) gAnim.totalTime = data.t.back();
    std::cout << "Loaded " << data.t.size() << " samples (total time = " << gAnim.totalTime << " s)\n";
    return data;
}

// ======================= Interpolation ===========================
osg::Vec3 interpolate(const std::vector<float> &tvec,
                      const std::vector<osg::Vec3> &vals, float t)
{
    if (tvec.empty()) return osg::Vec3();
    if (t <= tvec.front()) return vals.front();
    if (t >= tvec.back())  return vals.back();
    for (size_t i = 1; i < tvec.size(); ++i)
    {
        if (t < tvec[i])
        {
            float u = (t - tvec[i-1]) / (tvec[i]-tvec[i-1]);
            return vals[i-1]*(1.0f-u) + vals[i]*u;
        }
    }
    return vals.back();
}

// ======================= Orientation ===========================
static osg::Quat orientationFromTrajectory(const osg::Vec3 &pPrev,
                                           const osg::Vec3 &p,
                                           const osg::Vec3 &pNext)
{
    osg::Vec3 fwd = pNext - p;
    if (fwd.length2() < 1e-6) return osg::Quat();
    fwd.normalize();

    float yaw   = atan2f(fwd.y(), fwd.x());
    float pitch = asinf(-fwd.z());

    osg::Vec3 prevDir = p - pPrev;
    prevDir.normalize();
    osg::Vec3 turnAxis = prevDir ^ fwd;
    float turnStrength = turnAxis.length();
    float bankSign = (turnAxis.z() > 0.0f) ? -1.0f : 1.0f;
    float roll = bankSign * turnStrength * 1.8f;

    osg::Quat qYaw  (yaw  , osg::Vec3(0,0,1));
    osg::Quat qPitch(pitch, osg::Vec3(0,1,0));
    osg::Quat qRoll (roll , osg::Vec3(1,0,0));

    return qYaw * qPitch * qRoll;
}

// Helper to build orientation from velocity (for dynamic missile)
static osg::Quat orientationFromVelocity(const osg::Vec3& v, const osg::Vec3& up = osg::Vec3(0,0,-1))
{
    osg::Vec3 fwd = v; if (fwd.length2()<1e-8f) fwd.set(1,0,0); fwd.normalize();
    float yaw   = atan2f(fwd.y(), fwd.x());
    float pitch = asinf(-fwd.z());
    // roll heuristic = 0 for missile (no banking), keep it simple
    osg::Quat qYaw  (yaw  , osg::Vec3(0,0,1));
    osg::Quat qPitch(pitch, osg::Vec3(0,1,0));
    return qYaw * qPitch;
}

// ======================= Trail ===========================
class Trail : public osg::Referenced
{
public:
    Trail(size_t maxPoints = 2000, float minSegment = 0.2f)
        : _maxPoints(maxPoints), _minSegment(minSegment)
    {
        _verts = new osg::Vec3Array;
        _geom  = new osg::Geometry;
        _draw  = new osg::DrawArrays(GL_LINE_STRIP, 0, 0);
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
            size_t overflow = _verts->size() - _maxPoints;
            _verts->erase(_verts->begin(), _verts->begin() + overflow);
            _draw->setCount(_verts->size());
        }
    }

private:
    osg::ref_ptr<osg::Vec3Array> _verts;
    osg::ref_ptr<osg::Geometry>  _geom;
    osg::ref_ptr<osg::DrawArrays> _draw;
    osg::ref_ptr<osg::Geode>     _geode;
    size_t _maxPoints;
    float  _minSegment;
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

        // Position from file (keeps your logging/comparison consistent)
        osg::Vec3 pPrev = interpolate(data->t, data->aircraft, std::max(0.0f, t - DT_STEP));
        osg::Vec3 p     = interpolate(data->t, data->aircraft, t);
        osg::Vec3 pNext = interpolate(data->t, data->aircraft, std::min(gAnim.totalTime, t + DT_STEP));

        osg::Quat q = PLANE_BASIS * orientationFromTrajectory(pPrev, p, pNext);
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));

        if (trail.valid() && gAnim.running && gAnim.simTime > 0.0f && gAircraftOut.is_open())
        {
            osg::Vec3 bodyForward = q * (PLANE_BASIS * osg::Vec3(1, 0, 0));
            osg::Vec3 tail = p - bodyForward * gTailOffset;
            trail->add(tail);

            gAircraftOut << std::fixed << std::setprecision(3)
                         << t << " "
                         << p.x() << " " << p.y() << " " << p.z() << " "
                         << tail.x() << " " << tail.y() << " " << tail.z() << "\n";
        }

        traverse(mt.get(), nv);
    }

private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail>                trail;
    const TrajData *data;
};

class MissileCB : public osg::NodeCallback
{
public:
    MissileCB(osg::MatrixTransform *m, Trail *t, const TrajData *d, const TrajData *fighterData)
        : mt(m), trail(t), data(d), airData(fighterData) {}

    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        float t = gAnim.simTime;
        osg::Vec3 p;
        osg::Quat q;

        if (!gMissileParam.dynamicChase)
        {
            // ===== File / Scripted mode =====
            p  = interpolate(data->t, data->missile, t);
            osg::Vec3 p2 = interpolate(data->t, data->missile, std::min(gAnim.totalTime, t + DT_STEP));
            q  = MISSILE_BASIS * orientationFromTrajectory(p, p, p2);
        }
        else
        {
            // ===== Dynamic chase mode =====
            // Target: fighter position now + slight lookahead to avoid jitter
            osg::Vec3 target = interpolate(airData->t, airData->aircraft, std::min(gAnim.totalTime, t + 0.2f));

            // Time step in seconds (scaled by sim speed)
            float dt = DT_STEP * std::max(0.001f, gAnim.speed);

            // Desired direction
            osg::Vec3 dirDesired = target - gMissile.p;
            if (dirDesired.length2() < 1e-8f) dirDesired.set(1,0,0);
            dirDesired.normalize();

            // Current forward (from velocity)
            osg::Vec3 dirNow = gMissile.v; 
            if (dirNow.length2() < 1e-8f) dirNow.set(1,0,0);
            dirNow.normalize();

            // Limit turn by max angle per step
            float maxTurn = osg::DegreesToRadians(gMissileParam.turnRateDegPS) * dt;
            float dot = std::clamp(dirNow * dirDesired, -1.0f, 1.0f);
            float ang = acosf(dot);

            osg::Vec3 axis = dirNow ^ dirDesired;
            if (axis.length2() < 1e-12f) axis = osg::Vec3(0,0,1); // arbitrary
            axis.normalize();

            float step = std::min(maxTurn, ang);
            osg::Quat steer(step, axis);
            osg::Vec3 dirNew = steer * dirNow;
            dirNew.normalize();

            // Keep constant speed
            gMissile.v = dirNew * gMissileParam.speedMS;
            gMissile.p += gMissile.v * dt;

            p = gMissile.p;
            q = MISSILE_BASIS * orientationFromVelocity(gMissile.v);
        }

        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));

        if (trail.valid() && gAnim.running && gAnim.simTime > 0.0f && gMissileOut.is_open())
        {
            osg::Vec3 bodyForward = q * (MISSILE_BASIS * osg::Vec3(1, 0, 0));
            osg::Vec3 tail = p - bodyForward * 5.0f;
            trail->add(tail);

            gMissileOut << std::fixed << std::setprecision(3)
                        << t << " "
                        << p.x() << " " << p.y() << " " << p.z() << " "
                        << tail.x() << " " << tail.y() << " " << tail.z() << "\n";
        }

        traverse(mt.get(), nv);
    }

private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail>                trail;
    const TrajData *data;     // missile file
    const TrajData *airData;  // fighter file (for target position)
};

// ======================= Dynamic View Manager ===========================
enum class ViewPreset
{
    FighterSide,
    FighterFront,
    MissileTop,
    MissileRear,
    Custom
};

struct ViewInfo
{
    std::string label;
    ViewPreset  preset;
    osg::Vec3d  eye, center, up;
};

struct ViewEntry
{
    osg::ref_ptr<osgViewer::View> view;
    ViewInfo info;
};

class ViewManager
{
public:
    ViewManager(osgViewer::CompositeViewer* cv,
                osg::Group* root,
                osg::GraphicsContext* shareGC)
        : _cv(cv), _root(root), _sharedGC(shareGC) {}

    void createView(const ViewInfo& info, int w=800, int h=600)
    {
        osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
        traits->x = 120 + 40 * (_views.size() % 6);
        traits->y = 120 + 40 * (_views.size() % 6);
        traits->width = w;
        traits->height = h;
        traits->windowDecoration = true;
        traits->doubleBuffer = true;
        traits->sharedContext = _sharedGC;
        osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(traits.get());

        osg::ref_ptr<osgViewer::View> view = new osgViewer::View;
        view->setSceneData(_root.get());
        view->getCamera()->setGraphicsContext(gc.get());
        view->getCamera()->setViewport(new osg::Viewport(0,0,w,h));
        view->getCamera()->setProjectionMatrixAsPerspective(45.0, double(w)/double(h), 1.0, 10000.0);
        view->getCamera()->setClearColor(osg::Vec4(0.08f,0.08f,0.1f,1.0f));

        osg::ref_ptr<osgGA::OrbitManipulator> manip = new osgGA::OrbitManipulator;
        manip->setVerticalAxisFixed(false);
        manip->setHomePosition(info.eye, info.center, info.up);
        view->setCameraManipulator(manip.get());

        view->addEventHandler(new osgViewer::StatsHandler);

        _cv->addView(view.get());
        _views.push_back({view, info});

        std::cout << "Created view: " << info.label << "\n";
    }

    void destroyView(size_t idx)
    {
        if (idx >= _views.size()) return;
        _cv->removeView(_views[idx].view.get());
        std::cout << "Closed view: " << _views[idx].info.label << "\n";
        _views.erase(_views.begin() + idx);
    }

    size_t size() const { return _views.size(); }
    const std::vector<ViewEntry>& entries() const { return _views; }

    static ViewInfo presetInfo(ViewPreset p)
    {
        switch (p)
        {
            case ViewPreset::FighterSide:
                return {"Fighter Side", p, osg::Vec3(0,-80,  0), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)};
            case ViewPreset::FighterFront:
                return {"Fighter Front", p, osg::Vec3(60, 0,  0), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)};
            case ViewPreset::MissileTop:
                return {"Missile Top", p, osg::Vec3(0,-80,-30), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)};
            case ViewPreset::MissileRear:
                return {"Missile Rear", p, osg::Vec3(80, 0,-30), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)};
            default:
                return {"Custom", p, osg::Vec3(0,-50, 0), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)};
        }
    }

private:
    osg::observer_ptr<osgViewer::CompositeViewer> _cv;
    osg::observer_ptr<osg::Group>                 _root;
    osg::observer_ptr<osg::GraphicsContext>       _sharedGC;
    std::vector<ViewEntry>                        _views;
};

// ======================= ImGui UI (Main window) ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *a, Trail *m, ViewManager* vm)
        : ta(a), tm(m), _vm(vm) {}

protected:
    void drawUi() override
    {
        // === Trajectory Control ===
        ImGui::Begin("Trajectory Control");
        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.simTime = 0.0f;
            gAnim.running = false;
            if (ta.valid()) ta->clear();
            if (tm.valid()) tm->clear();
            if (gAircraftOut.is_open()) gAircraftOut << "# --- reset ---\n";
            if (gMissileOut.is_open())  gMissileOut  << "# --- reset ---\n";
            gMissile.reset();
            std::cout << "=== Trails cleared and animation reset ===\n";
        }
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 2.0f, "%.2f");
        ImGui::SliderFloat("Time (s)", &gAnim.simTime, 0.0f, gAnim.totalTime, "%.2f");

        ImGui::Separator();
        ImGui::Text("Missile Guidance");
        ImGui::Checkbox("Dynamic missile chase", &gMissileParam.dynamicChase);
        if (gMissileParam.dynamicChase)
        {
            ImGui::SliderFloat("Missile speed (m/s)", &gMissileParam.speedMS, 5.0f, 120.0f, "%.1f");
            ImGui::SliderFloat("Turn rate (deg/s)",  &gMissileParam.turnRateDegPS, 10.0f, 180.0f, "%.1f");
        }
        ImGui::End();

        // === Camera Manager ===
        ImGui::Begin("Camera Manager");

        static int presetIdx = 0;
        const char* presets[] = { "Fighter Side", "Fighter Front", "Missile Top", "Missile Rear", "Custom" };
        ImGui::Combo("Preset", &presetIdx, presets, IM_ARRAYSIZE(presets));

        static osg::Vec3 customEye(0,-60,0), customCenter(0,0,0), customUp(0,0,-1);
        if (presetIdx == 4) // Custom
        {
            ImGui::Separator();
            ImGui::Text("Custom Home");
            ImGui::Separator();
            ImGui::InputFloat3("Eye",    &customEye.x());
            ImGui::InputFloat3("Center", &customCenter.x());
            ImGui::InputFloat3("Up",     &customUp.x());
        }

        if (ImGui::Button("New Screen"))
        {
            ViewPreset p = (presetIdx==0)?ViewPreset::FighterSide:
                           (presetIdx==1)?ViewPreset::FighterFront:
                           (presetIdx==2)?ViewPreset::MissileTop:
                           (presetIdx==3)?ViewPreset::MissileRear:ViewPreset::Custom;

            ViewInfo info = (p==ViewPreset::Custom)
                          ? ViewInfo{"Custom", p, customEye, customCenter, customUp}
                          : ViewManager::presetInfo(p);

            if (_vm) _vm->createView(info, 900, 650);
        }

        ImGui::Separator();
        ImGui::Text("Active Screens: %zu", _vm ? _vm->size() : 0);
        if (_vm)
        {
            const auto& es = _vm->entries();
            for (size_t i=0; i<es.size(); ++i)
            {
                ImGui::PushID((int)i);
                ImGui::Text("%zu) %s", i+1, es[i].info.label.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Close")) _vm->destroyView(i);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

private:
    osg::observer_ptr<Trail> ta, tm;
    ViewManager* _vm = nullptr;
};

// ======================= Main ===========================
int main()
{
    osg::setNotifyLevel(osg::NotifySeverity::FATAL);

    const std::string baseDir  = "/home/murate/Documents/SwTrn/OsgPrc/osgtrn065/";
    const std::string trajFile = baseDir + "trajectory.txt";
    generateTrajectoryFile(trajFile);
    TrajData data = loadTrajectoryFile(trajFile);

    gAircraftOut.open(baseDir + "aircraft_traj_trail.txt");
    gMissileOut.open(baseDir + "missile_traj_trail.txt");
    if (gAircraftOut.is_open())
        gAircraftOut << "# time(s) ax ay az trailx traily trailz\n";
    if (gMissileOut.is_open())
        gMissileOut << "# time(s) mx my mz trailx traily trailz\n";

    // ===== Scene setup =====
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

    // Reset missile state once before running
    gMissile.reset();

    air->addUpdateCallback(new AircraftCB(air.get(), trailAircraft.get(), &data));
    mis->addUpdateCallback(new MissileCB(mis.get(), trailMissile.get(), &data, &data));
    root->addChild(air);
    root->addChild(mis);

    osg::ref_ptr<osg::Node> refAxes = osgDB::readNodeFile(dataPath + "axes.osgt");
    osg::ref_ptr<osg::MatrixTransform> refAxesXForm = new osg::MatrixTransform;
    refAxesXForm->addChild(refAxes);
    refAxesXForm->setMatrix(osg::Matrix::scale(2.0f, 2.0f, 2.0f));
    root->addChild(refAxesXForm.get());

    // ============================================================
    // ========== MAIN WINDOW: CompositeViewer + one View =========
    // ============================================================
    osg::ref_ptr<osgViewer::CompositeViewer> compViewer = new osgViewer::CompositeViewer;
    compViewer->setThreadingModel(osgViewer::CompositeViewer::SingleThreaded);
    compViewer->setRealizeOperation(new ImGuiInitOperation); // ImGui once

    // Create main window/GC
    const int winX = 100, winY = 100, winW = 1280, winH = 840;
    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
    traits->x = winX; traits->y = winY; traits->width = winW; traits->height = winH;
    traits->windowDecoration = true; traits->doubleBuffer = true; traits->sharedContext = 0;
    osg::ref_ptr<osg::GraphicsContext> mainGC = osg::GraphicsContext::createGraphicsContext(traits.get());

    // Main view with ImGui (control panel)
    osg::ref_ptr<osgViewer::View> mainView = new osgViewer::View;
    mainView->setSceneData(root.get());
    mainView->getCamera()->setGraphicsContext(mainGC.get());
    mainView->getCamera()->setViewport(new osg::Viewport(0,0,winW,winH));
    mainView->getCamera()->setProjectionMatrixAsPerspective(45.0, double(winW)/double(winH), 1.0, 10000.0);
    mainView->getCamera()->setClearColor(osg::Vec4(0.1f,0.1f,0.15f,1.0f));

    osg::ref_ptr<osgGA::OrbitManipulator> mainManip = new osgGA::OrbitManipulator;
    mainManip->setVerticalAxisFixed(false);
    mainManip->setHomePosition(osg::Vec3(0,-80,0), osg::Vec3(0,0,0), osg::Vec3(0,0,-1));
    mainView->setCameraManipulator(mainManip.get());

    compViewer->addView(mainView.get());

    // View Manager (uses mainGC as sharedContext for new windows)
    ViewManager viewMgr(compViewer.get(), root.get(), mainGC.get());

    // Attach ImGui panel to main view only
    mainView->addEventHandler(new ImGuiControl(trailAircraft.get(), trailMissile.get(), &viewMgr));

    // Optional: stats in main
    mainView->addEventHandler(new osgViewer::StatsHandler);

    // ============================================================
    // ========== Run Loop ========================================
    // ============================================================
    while (!compViewer->done())
        compViewer->frame();

    if (gAircraftOut.is_open()) gAircraftOut.close();
    if (gMissileOut.is_open())  gMissileOut.close();

    return 0;
}
