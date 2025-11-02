#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osgGA/TrackballManipulator>

int main(int argc, char** argv)
{
    // Root node of the scene
    osg::ref_ptr<osg::Group> root = new osg::Group;

    // Add axesNode under root
    osg::ref_ptr<osg::Node> axesNode = osgDB::readNodeFile("axes.osgt");
    if (!axesNode)
    {
        printf("Origin node not loaded, model not found\n");
        return 1;
    }
    root->addChild(axesNode);

    // Add a MatrixTransform under root
    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
    root->addChild(mt);

    // Add gliderNode under MatrixTransform
    osg::ref_ptr<osg::Node> gliderNode = osgDB::readNodeFile("glider.osg");
    if (!gliderNode)
    {
        printf("Glider node not loaded, model not found\n");
        return 1;
    }
    mt->addChild(gliderNode);

    // Create the viewer
    osgViewer::Viewer viewer;
    viewer.setSceneData(root);
    viewer.realize();

    // Attach a manipulator (it's usually done for us when we use viewer.run())
    osg::ref_ptr<osgGA::TrackballManipulator> tm = new osgGA::TrackballManipulator;
    viewer.setCameraManipulator(tm);

    int angle = 0;
    while (!viewer.done())
    {
        // Define the MatrixTransform's matrix
        osg::Matrix mRot  = osg::Matrix::rotate(osg::DegreesToRadians(double(angle)), osg::Z_AXIS);
        osg::Matrix mTrans = osg::Matrix::translate(5, 0, 0);
        osg::Matrix m = mRot * mTrans; // Translate then rotate
        mt->setMatrix(m);

        // Increment angle and wrap around for safety
        angle = (angle+1) % 360;

        viewer.frame();
    }

    return 0;
}