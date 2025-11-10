// osgtrn067.cpp
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
#include <osg/Camera>
#include <osg/StateSet>
#include <osgText/Text>
#include <osgDB/ReadFile>
#include <osgGA/OrbitManipulator>
#include <osgViewer/CompositeViewer>
#include <osgViewer/View>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/GraphicsWindow>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

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
    bool  running  = false;
    float simTime  = 0.0f;
    float totalTime= 10.0f;
    float speed    = 1.0f;
} gAnim;

const float DT_STEP = 0.1f;
float gTailOffset   = -14.0f;
const osg::Vec3 WORLD_UP(0, 0, -1);

struct MissileParams
{
    bool  dynamicChase  = false;
    float speedMS       = 30.0f;
    float turnRateDegPS = 60.0f;
} gMissileParam;

struct MissileState
{
    osg::Vec3 p = osg::Vec3(-110.0f, 0.0f, 0.0f);
    osg::Vec3 v = osg::Vec3(26.0f, 0.0f, 0.0f);
    void reset()
    {
        p.set(-110.0f, 0.0f, 0.0f);
        v = osg::Vec3(1, 0, 0) * gMissileParam.speedMS;
    }
} gMissile;

// ======================= Bases ===========================
const osg::Quat F14_MOD_BASIS(0, 0.731354, 0, 0.681998);
const osg::Quat MISSILE_BASIS(0.0, 0.0, 1.0, 0.0);
const osg::Quat PLANE_BASIS = F14_MOD_BASIS;

// ======================= Trajectories ===========================
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
    if (!data.t.empty())
        gAnim.totalTime = data.t.back();
    return data;
}

// ======================= Math helpers ===========================
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
            float u = (t - tvec[i - 1]) / (tvec[i] - tvec[i - 1]);
            return vals[i - 1] * (1 - u) + vals[i] * u;
        }
    }
    return vals.back();
}

osg::Quat orientationFromTrajectory(const osg::Vec3 &pPrev,
                                    const osg::Vec3 &p,
                                    const osg::Vec3 &pNext)
{
    osg::Vec3 fwd = pNext - p;
    if (fwd.length2() < 1e-6) return osg::Quat();
    fwd.normalize();
    float yaw = atan2f(fwd.y(), fwd.x());
    float pitch = asinf(-fwd.z());
    osg::Quat qYaw(yaw, osg::Vec3(0, 0, 1));
    osg::Quat qPitch(pitch, osg::Vec3(0, 1, 0));
    return qYaw * qPitch;
}

osg::Quat orientationFromVelocity(const osg::Vec3 &v)
{
    osg::Vec3 fwd = v;
    if (fwd.length2() < 1e-8) fwd.set(1,0,0);
    fwd.normalize();
    float yaw = atan2f(fwd.y(), fwd.x());
    float pitch = asinf(-fwd.z());
    osg::Quat qYaw(yaw, osg::Vec3(0,0,1));
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
        col->push_back(osg::Vec4(1, 1, 0.2, 1));
        _geom->setColorArray(col, osg::Array::BIND_OVERALL);

        osg::StateSet *ss = _geom->getOrCreateStateSet();
        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
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
    }

private:
    osg::ref_ptr<osg::Vec3Array> _verts;
    osg::ref_ptr<osg::Geometry>  _geom;
    osg::ref_ptr<osg::DrawArrays> _draw;
    osg::ref_ptr<osg::Geode>     _geode;
    size_t _maxPoints;
    float  _minSegment;
};

// ======================= Callbacks ===========================
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
            if (gAnim.simTime > gAnim.totalTime)
                gAnim.simTime = gAnim.totalTime;
        }
        float t = gAnim.simTime;
        osg::Vec3 pPrev = interpolate(data->t, data->aircraft, std::max(0.0f, t - DT_STEP));
        osg::Vec3 p     = interpolate(data->t, data->aircraft, t);
        osg::Vec3 pNext = interpolate(data->t, data->aircraft, std::min(gAnim.totalTime, t + DT_STEP));
        osg::Quat q = PLANE_BASIS * orientationFromTrajectory(pPrev, p, pNext);
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));

        if (trail.valid() && gAnim.running)
            trail->add(p - (q * (osg::Vec3(1, 0, 0) * gTailOffset)));

        traverse(mt.get(), nv);
    }
private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail>                trail;
    const TrajData*                         data;
};

class MissileCB : public osg::NodeCallback
{
public:
    MissileCB(osg::MatrixTransform *m, Trail *t, const TrajData *d, const TrajData *fighter)
        : mt(m), trail(t), data(d), airData(fighter) {}
    void operator()(osg::Node *, osg::NodeVisitor *nv) override
    {
        float t = gAnim.simTime;
        osg::Vec3 p;
        osg::Quat q;
        if (!gMissileParam.dynamicChase)
        {
            p = interpolate(data->t, data->missile, t);
            osg::Vec3 p2 = interpolate(data->t, data->missile, std::min(gAnim.totalTime, t + DT_STEP));
            q = MISSILE_BASIS * orientationFromTrajectory(p, p, p2);
        }
        else
        {
            osg::Vec3 target = interpolate(airData->t, airData->aircraft, std::min(gAnim.totalTime, t + 0.2f));
            float dt = DT_STEP * gAnim.speed;
            osg::Vec3 dirDesired = target - gMissile.p;
            if (dirDesired.length2() < 1e-8f) dirDesired.set(1,0,0);
            dirDesired.normalize();
            osg::Vec3 dirNow = gMissile.v; dirNow.normalize();
            float maxTurn = osg::DegreesToRadians(gMissileParam.turnRateDegPS) * dt;
            float dot = std::clamp(dirNow * dirDesired, -1.0f, 1.0f);
            float ang = acosf(dot);
            osg::Vec3 axis = dirNow ^ dirDesired;
            if (axis.length2() < 1e-8f) axis = osg::Vec3(0,0,1);
            axis.normalize();
            float step = std::min(maxTurn, ang);
            osg::Quat steer(step, axis);
            osg::Vec3 dirNew = steer * dirNow; dirNew.normalize();
            gMissile.v = dirNew * gMissileParam.speedMS;
            gMissile.p += gMissile.v * dt;
            p = gMissile.p;
            q = MISSILE_BASIS * orientationFromVelocity(gMissile.v);
        }
        mt->setMatrix(osg::Matrix::rotate(q) * osg::Matrix::translate(p));
        if (trail.valid() && gAnim.running)
            trail->add(p - (q * (osg::Vec3(1, 0, 0) * 5.0f)));
        traverse(mt.get(), nv);
    }
private:
    osg::observer_ptr<osg::MatrixTransform> mt;
    osg::observer_ptr<Trail>                trail;
    const TrajData*                         data;
    const TrajData*                         airData;
};

// ======================= HUD (overlay) ===========================
class HudTextCB : public osg::Callback
{
public:
    explicit HudTextCB(osgText::Text* text) : _text(text) {}
    bool run(osg::Object* obj, osg::Object* data) override
    {
        if (_text.valid())
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2)
               << "t: " << gAnim.simTime << " / " << gAnim.totalTime
               << "   speed: " << gAnim.speed
               << "   missile: " << (gMissileParam.dynamicChase ? "CHASE" : "SCRIPT");
            _text->setText(ss.str());
        }
        return traverse(obj, data);
    }
private:
    osg::observer_ptr<osgText::Text> _text;
};

osg::ref_ptr<osg::Geode> makeCrosshairGeode()
{
    osg::ref_ptr<osg::Geometry> g = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;

    const float r = 0.035f;
    v->push_back(osg::Vec3(-r, 0, 0)); v->push_back(osg::Vec3( r, 0, 0));
    v->push_back(osg::Vec3( 0,-r, 0)); v->push_back(osg::Vec3( 0, r, 0));

    g->setVertexArray(v.get());
    g->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 4));

    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
    c->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    g->setColorArray(c.get(), osg::Array::BIND_OVERALL);

    osg::StateSet* ss = g->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND,    osg::StateAttribute::ON);
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(2.0f);
    ss->setAttributeAndModes(lw, osg::StateAttribute::ON);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(g.get());
    return geode;
}

osg::ref_ptr<osg::Camera> createHudCamera()
{
    osg::ref_ptr<osg::Camera> cam = new osg::Camera;
    cam->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    cam->setClearMask(0);
    cam->setRenderOrder(osg::Camera::POST_RENDER);
    cam->setProjectionMatrixAsOrtho2D(-1.0, 1.0, -1.0, 1.0);
    cam->setViewMatrix(osg::Matrix::identity());

    osg::StateSet* ss = cam->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING,   osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

    // Crosshair
    cam->addChild(makeCrosshairGeode().get());

    // Text
    osg::ref_ptr<osgText::Text> txt = new osgText::Text;
    txt->setCharacterSize(0.06f);
    txt->setColor(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    txt->setPosition(osg::Vec3(-0.98f, 0.90f, 0.0f));
    txt->setAlignment(osgText::Text::LEFT_TOP); // OSG 3.6.x friendly
    txt->addUpdateCallback(new HudTextCB(txt.get()));

    osg::ref_ptr<osg::Geode> textGeode = new osg::Geode;
    textGeode->addDrawable(txt.get());
    cam->addChild(textGeode.get());

    return cam;
}

// ======================= ViewManager (with optional HUD) ===========================
enum class ViewPreset { FighterSide, FighterFront, MissileTop, MissileRear, Custom };
struct ViewInfo { std::string label; ViewPreset preset; osg::Vec3d eye, center, up; };
struct ViewEntry { osg::ref_ptr<osgViewer::View> view; ViewInfo info; };

class ViewManager
{
public:
    ViewManager(osgViewer::CompositeViewer *cv, osg::Group *root, osg::GraphicsContext *share)
        : _cv(cv), _root(root), _sharedGC(share) {}

    void createView(const ViewInfo &info, int w = 900, int h = 650, bool withHud = false)
    {
        osg::ref_ptr<osg::GraphicsContext::Traits> tr = new osg::GraphicsContext::Traits;
        tr->x = 100 + 40 * (_views.size() % 6);
        tr->y = 100 + 40 * (_views.size() % 6);
        tr->width = w; tr->height = h;
        tr->windowDecoration = true;
        tr->doubleBuffer = true;
        tr->sharedContext = _sharedGC;

        osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(tr.get());
        osg::ref_ptr<osgViewer::View> view = new osgViewer::View;

        osg::ref_ptr<osg::Group> perViewRoot = new osg::Group;
        perViewRoot->addChild(_root.get());
        if (withHud) perViewRoot->addChild(createHudCamera().get());
        view->setSceneData(perViewRoot.get());

        view->getCamera()->setGraphicsContext(gc.get());
        view->getCamera()->setViewport(new osg::Viewport(0, 0, w, h));
        view->getCamera()->setProjectionMatrixAsPerspective(45.0, double(w)/h, 1.0, 10000.0);
        view->getCamera()->setClearColor(osg::Vec4(0.08f, 0.08f, 0.1f, 1.0f));

        osg::ref_ptr<osgGA::OrbitManipulator> manip = new osgGA::OrbitManipulator;
        manip->setVerticalAxisFixed(false);
        manip->setHomePosition(info.eye, info.center, info.up);
        view->setCameraManipulator(manip.get());

        view->addEventHandler(new osgViewer::StatsHandler);
        _cv->addView(view.get());
        _views.push_back({view, info});
    }

    void duplicateView(osgViewer::View *src, const std::string &labelSuffix = " (Copy)", bool withHud = false)
    {
        if (!src) return;
        osg::Camera *cam = src->getCamera();
        if (!cam) return;

        osg::Matrixd viewM = cam->getViewMatrix();
        osg::Vec3d eye, center, up;
        viewM.getLookAt(eye, center, up);

        osg::ref_ptr<osg::GraphicsContext::Traits> tr = new osg::GraphicsContext::Traits;
        tr->x = 100 + 40 * (_views.size() % 6);
        tr->y = 100 + 40 * (_views.size() % 6);
        tr->width  = cam->getViewport()->width();
        tr->height = cam->getViewport()->height();
        tr->windowDecoration = true;
        tr->doubleBuffer = true;
        tr->sharedContext = _sharedGC;

        osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(tr.get());
        osg::ref_ptr<osgViewer::View> dup = new osgViewer::View;

        osg::ref_ptr<osg::Group> perViewRoot = new osg::Group;
        perViewRoot->addChild(_root.get());
        if (withHud) perViewRoot->addChild(createHudCamera().get());
        dup->setSceneData(perViewRoot.get());

        dup->getCamera()->setGraphicsContext(gc.get());
        dup->getCamera()->setViewport(new osg::Viewport(0, 0, tr->width, tr->height));
        dup->getCamera()->setProjectionMatrix(cam->getProjectionMatrix());
        dup->getCamera()->setViewMatrix(viewM);
        dup->getCamera()->setClearColor(cam->getClearColor());

        osg::ref_ptr<osgGA::OrbitManipulator> manip = new osgGA::OrbitManipulator;
        manip->setByMatrix(viewM);
        manip->setVerticalAxisFixed(false);
        dup->setCameraManipulator(manip.get());

        dup->addEventHandler(new osgViewer::StatsHandler);
        _cv->addView(dup.get());
        ViewInfo info{"Duplicated" + labelSuffix, ViewPreset::Custom, eye, center, up};
        _views.push_back({dup, info});
        std::cout << "Duplicated view created: " << info.label << "\n";
    }

    void destroyView(size_t i)
    {
        if (i >= _views.size()) return;
        _cv->removeView(_views[i].view.get());
        std::cout << "Closed view: " << _views[i].info.label << "\n";
        _views.erase(_views.begin() + i);
    }

    const std::vector<ViewEntry> &entries() const { return _views; }

private:
    osg::observer_ptr<osgViewer::CompositeViewer> _cv;
    osg::observer_ptr<osg::Group> _root;
    osg::observer_ptr<osg::GraphicsContext> _sharedGC;
    std::vector<ViewEntry> _views;
};

// ======================= ImGuiControl ===========================
class ImGuiControl : public OsgImGuiHandler
{
public:
    ImGuiControl(Trail *a, Trail *m, ViewManager *vm) : ta(a), tm(m), _vm(vm) {}

protected:
    void drawUi() override
    {
        ImGui::Begin("Trajectory Control");
        if (ImGui::Button(gAnim.running ? "Stop" : "Start")) gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.running = false;
            gAnim.simTime = 0;
            if (ta.valid()) ta->clear();
            if (tm.valid()) tm->clear();
            gMissile.reset();
        }

        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 2.0f, "%.2f");
        ImGui::SliderFloat("Time",  &gAnim.simTime, 0, gAnim.totalTime, "%.2f");
        ImGui::Separator();
        ImGui::Text("Missile Guidance");
        ImGui::Checkbox("Dynamic missile chase", &gMissileParam.dynamicChase);
        if (gMissileParam.dynamicChase)
        {
            ImGui::SliderFloat("Missile speed", &gMissileParam.speedMS, 5, 120, "%.1f");
            ImGui::SliderFloat("Turn rate",     &gMissileParam.turnRateDegPS, 10, 180, "%.1f");
        }
        ImGui::End();

        ImGui::Begin("Camera Manager");
        static int presetIdx = 0;
        const char *presets[] = {"Fighter Side", "Fighter Front", "Missile Top", "Missile Rear", "Custom"};
        ImGui::Combo("Preset", &presetIdx, presets, IM_ARRAYSIZE(presets));
        static osg::Vec3 eye(0, -60, 0), center(0, 0, 0), up(0, 0, -1);
        if (presetIdx == 4)
        {
            ImGui::InputFloat3("Eye",    &eye.x());
            ImGui::InputFloat3("Center", &center.x());
            ImGui::InputFloat3("Up",     &up.x());
        }

        static bool newScreenWithHud = false; // default OFF: no HUD on new screens
        ImGui::Checkbox("New Screen: with HUD", &newScreenWithHud);

        if (ImGui::Button("New Screen"))
        {
            ViewInfo info;
            switch (presetIdx)
            {
                case 0: info = {"Fighter Side",  ViewPreset::FighterSide, osg::Vec3(0, -80, 0),  osg::Vec3(0,0,0), osg::Vec3(0,0,-1)}; break;
                case 1: info = {"Fighter Front", ViewPreset::FighterFront, osg::Vec3(60, 0, 0),  osg::Vec3(0,0,0), osg::Vec3(0,0,-1)}; break;
                case 2: info = {"Missile Top",   ViewPreset::MissileTop,   osg::Vec3(0, -80, -30),osg::Vec3(0,0,0), osg::Vec3(0,0,-1)}; break;
                case 3: info = {"Missile Rear",  ViewPreset::MissileRear,  osg::Vec3(80, 0, -30), osg::Vec3(0,0,0), osg::Vec3(0,0,-1)}; break;
                default: info = {"Custom", ViewPreset::Custom, eye, center, up}; break;
            }
            if (_vm) _vm->createView(info, 900, 650, newScreenWithHud);
        }

        ImGui::SameLine();
        static bool dupWithHud = false; // default OFF: duplicated screens have no HUD
        ImGui::Checkbox("Duplicate: with HUD", &dupWithHud);

        if (ImGui::Button("Duplicate Current"))
        {
            if (_vm && !_vm->entries().empty())
                _vm->duplicateView(_vm->entries().back().view.get(), " (Copy)", dupWithHud);
        }

        ImGui::Separator();
        if (_vm)
        {
            const auto &es = _vm->entries();
            ImGui::Text("Active Screens: %zu", es.size());
            for (size_t i = 0; i < es.size(); ++i)
            {
                ImGui::PushID((int)i);
                ImGui::Text("%zu) %s", i + 1, es[i].info.label.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Close")) _vm->destroyView(i);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

private:
    osg::observer_ptr<Trail> ta, tm;
    ViewManager *_vm = nullptr;
};

// ======================= Main ===========================
int main()
{
    const std::string baseDir  = "/home/murate/Documents/SwTrn/OsgPrc/osgtrn067/";
    const std::string trajFile = baseDir + "trajectory.txt";
    generateTrajectoryFile(trajFile);
    TrajData data = loadTrajectoryFile(trajFile);

    osg::ref_ptr<osg::Group> root = new osg::Group;
    osg::ref_ptr<Trail> tA = new Trail, tM = new Trail;

    osg::ref_ptr<osg::Node> f14  = osgDB::readRefNodeFile("/home/murate/Documents/SwTrn/OsgPrc/OpenSceneGraph-Data/F-14-low-poly-axes-modified.ac");
    osg::ref_ptr<osg::Node> aim9 = osgDB::readRefNodeFile("/home/murate/Documents/SwTrn/OsgPrc/OpenSceneGraph-Data/AIM-9L.ac");

    osg::ref_ptr<osg::MatrixTransform> air = new osg::MatrixTransform;
    air->addChild(f14);
    osg::ref_ptr<osg::MatrixTransform> mis = new osg::MatrixTransform;
    mis->addChild(aim9);

    air->addUpdateCallback(new AircraftCB(air.get(), tA.get(), &data));
    mis->addUpdateCallback(new MissileCB(mis.get(), tM.get(), &data, &data));

    root->addChild(air);
    root->addChild(mis);
    root->addChild(tA->geode());
    root->addChild(tM->geode());

    osg::ref_ptr<osgViewer::CompositeViewer> cv = new osgViewer::CompositeViewer;
    cv->setThreadingModel(osgViewer::CompositeViewer::SingleThreaded);
    cv->setRealizeOperation(new ImGuiInitOperation);

    osg::ref_ptr<osg::GraphicsContext::Traits> tr = new osg::GraphicsContext::Traits;
    tr->x = 100; tr->y = 100; tr->width = 1280; tr->height = 840;
    tr->windowDecoration = true;
    tr->doubleBuffer = true;
    osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(tr.get());

    // Main View: WITH HUD
    osg::ref_ptr<osgViewer::View> mainV = new osgViewer::View;
    osg::ref_ptr<osg::Group> mainPerViewRoot = new osg::Group;
    mainPerViewRoot->addChild(root.get());
    mainPerViewRoot->addChild(createHudCamera().get()); // only main has HUD
    mainV->setSceneData(mainPerViewRoot.get());

    mainV->getCamera()->setGraphicsContext(gc.get());
    mainV->getCamera()->setViewport(new osg::Viewport(0, 0, 1280, 840));
    mainV->getCamera()->setProjectionMatrixAsPerspective(45.0, 1280.0/840.0, 1.0, 10000.0);
    mainV->getCamera()->setClearColor(osg::Vec4(0.1f, 0.1f, 0.15f, 1.0f));

    osg::ref_ptr<osgGA::OrbitManipulator> manip = new osgGA::OrbitManipulator;
    manip->setVerticalAxisFixed(false);
    manip->setHomePosition(osg::Vec3(0, -80, 0), osg::Vec3(0, 0, 0), osg::Vec3(0, 0, -1));
    mainV->setCameraManipulator(manip.get());
    cv->addView(mainV.get());

    ViewManager vm(cv.get(), root.get(), gc.get());
    mainV->addEventHandler(new ImGuiControl(tA.get(), tM.get(), &vm));

    // Run
    while (!cv->done())
        cv->frame();
    return 0;
}