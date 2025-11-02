#include <osg/Group>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <iostream>

int main()
{
    osgViewer::Viewer viewer;

    // Path to OSG sample data (Debian installs here)
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(dataPath + "F-14-low-poly.ac");
    if (!model)
    {
        std::cerr << "Failed to load model from: " << dataPath << std::endl;
        return 1;
    }

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(model);

    viewer.setSceneData(root);
    viewer.setUpViewInWindow(100, 100, 600, 600);

    return viewer.run();
}
