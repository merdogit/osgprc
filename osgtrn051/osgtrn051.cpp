// ==== FIXED: GLEW must come first ====
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include <GL/glew.h>

// ==== Now include your own headers ====
#include "CameraController.hpp"
#include "FollowOrbitManipulator.hpp"
#include "ManipulatorControlPanel.hpp"
#include "ImGuiSetup.hpp"
#include "OsgImGuiHandler.hpp"
#include "CommonFunctions"

#include <osg/Camera>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/OrbitManipulator>
#include <osgViewer/Viewer>


int main(int argc, char** argv)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::MatrixTransform> trans = new osg::MatrixTransform;
    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");
    trans->addUpdateCallback(osgCookBook::createAnimationPathCallback(100.0f, 20.0f));
    trans->addChild(model.get());

    osg::ref_ptr<osg::MatrixTransform> terrain = new osg::MatrixTransform;
    terrain->addChild(osgDB::readNodeFile(dataPath + "lz.osg"));
    terrain->setMatrix(osg::Matrix::translate(0.0, 0.0, -200.0));

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(trans.get());
    root->addChild(terrain.get());

    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 900, 700);
    viewer.setSceneData(root.get());
    viewer.setRealizeOperation(new ImGuiInitOperation());

    // ---- Unified camera control ----
    CameraController cameraCtrl(trans.get());
    cameraCtrl.attach(viewer);

    // ---- Optional ImGui control panel ----
    osg::ref_ptr<ManipulatorControlPanel> imguiPanel =
        new ManipulatorControlPanel(cameraCtrl.getKeySwitch(), cameraCtrl.getFollow());
    viewer.addEventHandler(imguiPanel.get());

    int result = viewer.run();

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    return result;
}