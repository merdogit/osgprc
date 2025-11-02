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

#define ANSI_RESET "\e[0;0m]"
#define ANSI_CYAN "\e[0;36m"

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

struct AnimationState
{
    bool running = false;
    float t = 0.0f;
    float speed = 0.25f;
} gAnim;

const osg::Vec3 WORLD_UP(0, 0, -1);

osg::ref_ptr<osg::Geode> createAxes(float len = 10.0f)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array();

    v->push_back({0, 0, 0}); v->push_back({-len, 0, 0}); c->push_back({1,0,0,1}); c->push_back({1,0,0,1});
    v->push_back({0, 0, 0}); v->push_back({0, 0, -len}); c->push_back({0,1,0,1}); c->push_back({0,1,0,1});
    v->push_back({0, 0, 0}); v->push_back({0, -len, 0}); c->push_back({0,0,1,1}); c->push_back({0,0,1,1});

    geom->setVertexArray(v);
    geom->setColorArray(c, osg::Array::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, v->size()));
    geom->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(3.0f), osg::StateAttribute::ON);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);
    return geode;
}

// -------------------- ImGui Motion Control --------------------
class ImGuiControl : public OsgImGuiHandler
{
protected:
    void drawUi() override
    {
        ImGui::Begin("Motion Controller");
        if (ImGui::Button(gAnim.running ? "Stop" : "Start"))
            gAnim.running = !gAnim.running;
        ImGui::SliderFloat("Speed", &gAnim.speed, 0.05f, 1.0f, "%.2f");
        ImGui::End();
    }
};

// -------------------- Light Control --------------------
class LightControl : public OsgImGuiHandler
{
public:
    LightControl(osg::LightSource* lightSrc, osg::MatrixTransform* symbolXform)
        : _lightSrc(lightSrc), _symbolXform(symbolXform)
    {
        _pos = osg::Vec3(0.0f, 50.0f, -80.0f);
        _dir = osg::Vec3(0.0f, 0.0f, 1.0f);
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
            light->setPosition(osg::Vec4(_dir, 0.0f));
        else
            light->setPosition(osg::Vec4(_pos, 1.0f));

        light->setAmbient(_ambient);
        light->setDiffuse(_diffuse);
        light->setSpecular(_specular);
        if (_enabled)
            _lightSrc->setLocalStateSetModes(osg::StateAttribute::ON);
        else
            _lightSrc->setLocalStateSetModes(osg::StateAttribute::OFF);

        if (_symbolXform.valid())
        {
            osg::Matrix M;
            if (_directional)
            {
                osg::Vec3 dir = _dir; dir.normalize();
                osg::Matrix R = osg::Matrix::rotate(osg::Vec3(0, 0, -1), dir);
                M = R * osg::Matrix::translate(_pos);
            }
            else
                M = osg::Matrix::translate(_pos);
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

// -------------------- Create Sky Dome --------------------
osg::ref_ptr<osg::Geode> createSkyDome(float radius = 500.0f, int segments = 32)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> cols = new osg::Vec4Array();

    // Create hemisphere pointing downward (since -Z is up)
    for (int i = 0; i <= segments / 2; ++i)
    {
        float theta = osg::PI * 0.5f * (float)i / (segments / 2); // 0..π/2
        float z = -radius * sin(theta);
        float r = radius * cos(theta);
        for (int j = 0; j <= segments; ++j)
        {
            float phi = 2.0f * osg::PI * (float)j / segments;
            float x = r * cos(phi);
            float y = r * sin(phi);
            verts->push_back(osg::Vec3(x, y, z - 200.0f)); // center at z=-200
            float t = (float)i / (segments / 2);
            osg::Vec4 color(0.3f + 0.2f*t, 0.5f + 0.3f*t, 1.0f, 1.0f);
            cols->push_back(color);
        }
    }

    geom->setVertexArray(verts);
    geom->setColorArray(cols, osg::Array::BIND_PER_VERTEX);

    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(GL_TRIANGLES);
    int colsCount = segments + 1;
    for (int i = 0; i < segments / 2; ++i)
    {
        for (int j = 0; j < segments; ++j)
        {
            int a = i * colsCount + j;
            int b = (i + 1) * colsCount + j;
            int c = (i + 1) * colsCount + (j + 1);
            int d = i * colsCount + (j + 1);
            indices->push_back(a); indices->push_back(b); indices->push_back(d);
            indices->push_back(b); indices->push_back(c); indices->push_back(d);
        }
    }
    geom->addPrimitiveSet(indices);

    osg::ref_ptr<osg::Geode> dome = new osg::Geode();
    dome->addDrawable(geom);
    osg::StateSet* ss = dome->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    return dome;
}

// -------------------- Main --------------------
int main(int, char**)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    // --- Light setup ---
    osg::ref_ptr<osg::Light> light = new osg::Light;
    light->setLightNum(0);
    light->setPosition(osg::Vec4(0.0f, 0.0f, 1.0f, 0.0f));
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

    // --- Axes ---
    root->addChild(createAxes(20.0f));

    // --- Sky Dome ---
    osg::ref_ptr<osg::Geode> sky = createSkyDome(500.0f, 48);
    root->addChild(sky);

    // --- Viewer ---
    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 1000, 700));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiControl());
    viewer.addEventHandler(new LightControl(lightSrc.get(), lightSymbolXform.get()));

    return viewer.run();
}
