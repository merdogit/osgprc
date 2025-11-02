// Kamera yerlesim kalibrasyonu

#include <osg/Geometry>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/GUIEventHandler>
#include <osg/Notify>
#include <iostream>


int main(int argc, char **argv)
{
    osg::notify(osg::WARN);

    osg::ref_ptr<osg::Group> root = new osg::Group();

    // Load cessna model
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(dataPath + "F-14-low-poly-no-land-gear.ac");
    if (!model)
    {
        std::cerr << "Error: Could not load model\n";
        return 1;
    }

    osg::ref_ptr<osg::MatrixTransform> modelXform = new osg::MatrixTransform;
    modelXform->addChild(model.get());
    root->addChild(modelXform.get());


    modelXform->setMatrix(osg::Matrix::rotate(osg::Quat(-0.00612029, -0.700665, 0.713439, 0.00601263)));

    osgViewer::Viewer viewer;
    viewer.setSceneData(root.get());

    // Camera manipulator
    osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;

    manip->setTrackNode(modelXform);
    manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER);
    manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);

    manip->setHomePosition(
        osg::Vec3(-50.0f, 0.0f, -10.0f), // offset (behind & above)
        osg::Vec3(0.0f, 0.0f, 0.0f),    // look at the model center
        osg::Vec3(0.0f, 0.0f, -1.0f)     // up direction
    );

    viewer.setCameraManipulator(manip.get());

 
    viewer.setUpViewInWindow(700, 50, 800, 600);

    return viewer.run();
}