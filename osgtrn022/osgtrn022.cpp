#include <iostream>
#include <cmath>

#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Point>
#include <osg/StateSet>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>
#include <osgGA/TrackballManipulator>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include "OsgImGuiHandler.hpp"

// ======================
// ImGui + OSG Integration
// ======================
class ImGuiInitOperation : public osg::Operation
{
public:
    ImGuiInitOperation()
        : osg::Operation("ImGuiInitOperation", false)
    {
    }

    void operator()(osg::Object *object) override
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;

        if (!ImGui_ImplOpenGL3_Init())
        {
            std::cout << "ImGui_ImplOpenGL3_Init() failed\n";
        }
    }
};

// ======================
// Cubic Grid Generator (points)
// ======================
class GridGenerator
{
public:
    static osg::ref_ptr<osg::Geode> createCubicGrid(float spacing, int halfCount = 5)
    {
        osg::ref_ptr<osg::Geode> geode = new osg::Geode;
        osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
        osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;

        // Generate cubic grid of points
        for (int x = -halfCount; x <= halfCount; ++x)
        {
            for (int y = -halfCount; y <= halfCount; ++y)
            {
                for (int z = -halfCount; z <= halfCount; ++z)
                {
                    vertices->push_back(osg::Vec3(x * spacing, y * spacing, z * spacing));
                }
            }
        }

        geom->setVertexArray(vertices);
        geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, vertices->size()));

        // Point color
        osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(0.2f, 0.7f, 1.0f, 1.0f)); // cyan-blue
        geom->setColorArray(colors, osg::Array::BIND_OVERALL);

        // Visual settings
        osg::ref_ptr<osg::StateSet> ss = geom->getOrCreateStateSet();

        // Disable lighting (so color is not affected by light)
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        // Set point size (make visible)
        osg::ref_ptr<osg::Point> point = new osg::Point;
        point->setSize(3.0f);
        ss->setAttributeAndModes(point, osg::StateAttribute::ON);

        // Optional: disable depth test for always-visible points (can comment if not wanted)
        // ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

        geode->addDrawable(geom);
        return geode;
    }
};

// ======================
// ImGui Controller
// ======================
class ImGuiDemo : public OsgImGuiHandler
{
public:
    ImGuiDemo(osg::Group *root)
        : _root(root), _gridSpacing(1.0f), _lastSpacing(1.0f)
    {
        _gridNode = GridGenerator::createCubicGrid(_gridSpacing);
        _root->addChild(_gridNode);
    }

protected:
    void drawUi() override
    {
        ImGui::Begin("3D Grid Controller");
        ImGui::Text("Adjust cubic grid spacing:");
        ImGui::SliderFloat("Spacing", &_gridSpacing, 0.5f, 5.0f, "%.1f");

        if (std::fabs(_gridSpacing - _lastSpacing) > 0.001f)
        {
            _root->removeChild(_gridNode);
            _gridNode = GridGenerator::createCubicGrid(_gridSpacing);
            _root->addChild(_gridNode);
            _lastSpacing = _gridSpacing;
        }

        ImGui::End();
    }

private:
    osg::observer_ptr<osg::Group> _root;
    osg::ref_ptr<osg::Node> _gridNode;
    float _gridSpacing;
    float _lastSpacing;
};

// ======================
// Main
// ======================
int main(int argc, char **argv)
{
    osg::ref_ptr<osg::Group> root = new osg::Group;

    osgViewer::Viewer viewer;
    viewer.apply(new osgViewer::SingleWindow(100, 100, 800, 600));
    viewer.setRealizeOperation(new ImGuiInitOperation);
    viewer.setCameraManipulator(new osgGA::TrackballManipulator);
    viewer.setSceneData(root);

    // Add ImGui handler
    viewer.addEventHandler(new ImGuiDemo(root));

    return viewer.run();
}