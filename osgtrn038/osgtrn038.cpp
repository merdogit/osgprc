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
#include <osg/PolygonOffset>
#include <osg/ShapeDrawable>
#include <osg/Light>
#include <osg/LightSource>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "OsgImGuiHandler.hpp"

// ======================= ANSI Color Codes ===========================
#define ANSI_RESET "\e[0;0m]"
#define ANSI_CYAN "\e[0;36m"

// ======================= ImGui Init ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object* object) override
    {
        auto* gc = dynamic_cast<osg::GraphicsContext*>(object);
        if (!gc) return;
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
const osg::Quat MISSILE_BASIS(0, 0, 1, 0);

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

// ======================= Debug Axes ===========================
osg::ref_ptr<osg::Geode> createAxes(float len = 10.0f)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array();

    // X (red)
    v->push_back({0, 0, 0}); v->push_back({-len, 0, 0});
    c->push_back({1, 0, 0, 1}); c->push_back({1, 0, 0, 1});
    // Y (green)
    v->push_back({0, 0, 0}); v->push_back({0, 0, -len});
    c->push_back({0, 1, 0, 1}); c->push_back({0, 1, 0, 1});
    // Z (blue)
    v->push_back({0, 0, 0}); v->push_back({0, -len, 0});
    c->push_back({0, 0, 1, 1}); c->push_back({0, 0, 1, 1});

    geom->setVertexArray(v);
    geom->setColorArray(c, osg::Array::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, v->size()));
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(3.0f);
    geom->getOrCreateStateSet()->setAttributeAndModes(lw, osg::StateAttribute::ON);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);
    return geode;
}

// ======================= ImGui Motion Control ===========================
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("Motion Controller");
        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            gAnim.t = 0.0f; gAnim.running = false;
            std::cout << ANSI_CYAN << "=== Reset motion & trails ===" << ANSI_RESET << std::endl;
        }
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::SliderFloat("t (timeline)", &gAnim.t, 0.0f, 1.0f, "%.3f");
        ImGui::End();
    }
};

// ======================= Light Control ===========================
class LightControl : public OsgImGuiHandler
{
public:
    LightControl(osg::LightSource* lightSrc, osg::MatrixTransform* symbolXform)
        : _lightSrc(lightSrc), _symbolXform(symbolXform)
    {
        _pos = osg::Vec3(0.0f, 50.0f, -80.0f);
        _dir = osg::Vec3(0.0f, 0.0f, 1.0f); // upward (sky is -Z)
        _ambient = osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f);
        _diffuse = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
        _specular = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
        _directional = true;
        _enabled = true;
    }

protected:
    void drawUi() override
    {
        ImGui::Begin("Light Controls");
        ImGui::Checkbox("Enable Light", &_enabled);
        ImGui::Checkbox("Directional (Sunlight)", &_directional);
        ImGui::SliderFloat3("Position (XYZ)", _pos.ptr(), -200.0f, 200.0f, "%.1f");
        ImGui::SliderFloat3("Direction", _dir.ptr(), -1.0f, 1.0f, "%.2f");
        ImGui::ColorEdit3("Ambient", _ambient.ptr());
        ImGui::ColorEdit3("Diffuse", _diffuse.ptr());
        ImGui::ColorEdit3("Specular", _specular.ptr());

        osg::Light* light = _lightSrc->getLight();
        if (_directional)
            light->setPosition(osg::Vec4(_dir, 0.0f)); // directional
        else
            light->setPosition(osg::Vec4(_pos, 1.0f)); // positional

        light->setAmbient(_ambient);
        light->setDiffuse(_diffuse);
        light->setSpecular(_specular);

        if (_enabled)
            _lightSrc->setLocalStateSetModes(osg::StateAttribute::ON);
        else
            _lightSrc->setLocalStateSetModes(osg::StateAttribute::OFF);

        // === Update the light symbol transform ===
        if (_symbolXform.valid())
        {
            osg::Matrix M;
            if (_directional)
            {
                osg::Vec3 dir = _dir;
                dir.normalize();
                osg::Matrix R = osg::Matrix::rotate(osg::Vec3(0, 0, -1), dir);
                M = R * osg::Matrix::translate(_pos);
            }
            else
            {
                M = osg::Matrix::translate(_pos);
            }
            _symbolXform->setMatrix(M);
        }

        ImGui::End();
    }

private:
    osg::observer_ptr<osg::LightSource> _lightSrc;
    osg::observer_ptr<osg::MatrixTransform> _symbolXform;
    osg::Vec3 _pos, _dir;
    osg::Vec4 _ambient, _diffuse, _specular;
    bool _directional, _enabled;
};

// ======================= Main ===========================
int main(int, char**)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    // --- Light setup ---
    osg::ref_ptr<osg::Light> light = new osg::Light;
    light->setLightNum(0);
    light->setPosition(osg::Vec4(0.0f, 0.0f, 1.0f, 0.0f)); // upward
    light->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    light->setDiffuse(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    light->setSpecular(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

    osg::ref_ptr<osg::LightSource> lightSrc = new osg::LightSource;
    lightSrc->setLight(light);
    root->addChild(lightSrc);

    // --- Light visualization symbol ---
    osg::ref_ptr<osg::Geode> lightSymbol = new osg::Geode();

    osg::ref_ptr<osg::ShapeDrawable> bulb = new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0, 0, 0), 2.0f));
    bulb->setColor(osg::Vec4(1.0f, 1.0f, 0.7f, 1.0f));
    lightSymbol->addDrawable(bulb);

    osg::ref_ptr<osg::ShapeDrawable> cone = new osg::ShapeDrawable(new osg::Cone(osg::Vec3(0, 0, -6.0f), 2.0f, 8.0f));
    cone->setColor(osg::Vec4(1.0f, 1.0f, 0.6f, 0.4f));
    lightSymbol->addDrawable(cone);

    osg::StateSet* symbolSS = lightSymbol->getOrCreateStateSet();
    symbolSS->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    symbolSS->setMode(GL_BLEND, osg::StateAttribute::ON);
    symbolSS->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::ref_ptr<osg::MatrixTransform> lightSymbolXform = new osg::MatrixTransform;
    lightSymbolXform->addChild(lightSymbol);
    root->addChild(lightSymbolXform);

    // --- Reference axes ---
    osg::ref_ptr<osg::Geode> axes = createAxes(20.0f);
    root->addChild(axes);

    // --- Viewer ---
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.addEventHandler(new LightControl(lightSrc.get(), lightSymbolXform.get()));

    return viewer.run();
}