#include <iostream>

#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/MatrixTransform>
#include <osg/LineWidth>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include "OsgImGuiHandler.hpp"

// ======================= ImGui Initialization ===========================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation() : osg::Operation("ImGuiInitOperation", false) {}
    void operator()(osg::Object *object) override
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context) return;
        if (!ImGui_ImplOpenGL3_Init())
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
    }
};

// ======================= Grid Generator ===========================
osg::ref_ptr<osg::Geode> createGridPoints(int gridCount, float spacing)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

    for (int x = -gridCount; x <= gridCount; ++x)
        for (int y = -gridCount; y <= gridCount; ++y)
            for (int z = -gridCount; z <= gridCount; ++z)
                vertices->push_back(osg::Vec3(x * spacing, y * spacing, z * spacing));

    geom->setVertexArray(vertices);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, vertices->size()));

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(osg::Vec4(1.0, 1.0, 1.0, 1.0)); // white points
    geom->setColorArray(colors, osg::Array::BIND_OVERALL);

    osg::ref_ptr<osg::StateSet> ss = geode->getOrCreateStateSet();
    osg::ref_ptr<osg::Point> pointSize = new osg::Point(3.0f);
    ss->setAttribute(pointSize);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

    geode->addDrawable(geom);
    return geode;
}

// ======================= Axes Generator ===========================
osg::ref_ptr<osg::Geode> createAxes(float length)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    // X-axis (red)
    vertices->push_back(osg::Vec3(0, 0, 0));
    vertices->push_back(osg::Vec3(length, 0, 0));
    colors->push_back(osg::Vec4(1, 0, 0, 1));
    colors->push_back(osg::Vec4(1, 0, 0, 1));

    // Y-axis (green)
    vertices->push_back(osg::Vec3(0, 0, 0));
    vertices->push_back(osg::Vec3(0, length, 0));
    colors->push_back(osg::Vec4(0, 1, 0, 1));
    colors->push_back(osg::Vec4(0, 1, 0, 1));

    // Z-axis (blue)
    vertices->push_back(osg::Vec3(0, 0, 0));
    vertices->push_back(osg::Vec3(0, 0, length));
    colors->push_back(osg::Vec4(0, 0, 1, 1));
    colors->push_back(osg::Vec4(0, 0, 1, 1));

    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, vertices->size()));

    osg::ref_ptr<osg::StateSet> ss = geode->getOrCreateStateSet();
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(4.0f);
    ss->setAttribute(lw);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON); // enable depth for proper 3D rendering

    geode->addDrawable(geom);
    return geode;
}

// ======================= ImGui UI Handler ===========================
class ImGuiDemo : public OsgImGuiHandler
{
public:
    ImGuiDemo(osg::Group* root)
        : _root(root)
    {
        _gridCount = 5;
        _spacing = 1.0f;
        rebuildScene();
    }

protected:
    void drawUi() override
    {
        ImGui::Begin("Grid Control");
        ImGui::SliderInt("Grid count", &_gridCount, 1, 20);
        ImGui::SliderFloat("Spacing", &_spacing, 0.5f, 5.0f);
        if (ImGui::Button("Update Grid"))
            rebuildScene();
        ImGui::End();
    }

    void rebuildScene()
    {
        std::cout << "rebuildScene()\n";

        if (_gridTransform.valid())
            _root->removeChild(_gridTransform);

        _gridTransform = new osg::MatrixTransform();

        // Add axes first so they are always visible at origin
        _gridTransform->addChild(createAxes((_gridCount + 2) * _spacing));

        // Add cubic grid points
        _gridTransform->addChild(createGridPoints(_gridCount, _spacing));

        _root->addChild(_gridTransform);
    }

private:
    osg::observer_ptr<osg::Group> _root;
    osg::ref_ptr<osg::MatrixTransform> _gridTransform;
    int _gridCount;
    float _spacing;
};

// ======================= Main ===========================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 800, 600));
    viewer.setSceneData(root);
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.addEventHandler(new ImGuiDemo(root));
    viewer.setLightingMode(osg::View::NO_LIGHT);

    return viewer.run();
}