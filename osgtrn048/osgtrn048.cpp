#include <osg/Camera>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/TrackballManipulator>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/OrbitManipulator>
#include <osgViewer/Viewer>
#include <iostream>

#include "CommonFunctions"

// --- Helper: compute world matrix ---
static osg::Matrix worldMatrixOf(osg::Node* node)
{
    if (!node) return osg::Matrix::identity();
    const auto& paths = node->getParentalNodePaths();
    if (paths.empty()) return osg::Matrix::identity();
    return osg::computeLocalToWorld(paths.front());
}

// --- FollowOrbitManipulator with offset and orientation alignment ---
class FollowOrbitManipulator : public osgGA::OrbitManipulator
{
public:
    explicit FollowOrbitManipulator(osg::Node* target)
        : _target(target), _offset(0.0, -60.0, 25.0), _alignYaw(true) {}

    void setTarget(osg::Node* n) { _target = n; }
    void setOffset(const osg::Vec3d& off) { _offset = off; }
    void setAlignYaw(bool enable) { _alignYaw = enable; }

    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override
    {
        bool handled = osgGA::OrbitManipulator::handle(ea, aa);

        if (_target.valid() && ea.getEventType() == osgGA::GUIEventAdapter::FRAME)
        {
            osg::Matrix world = worldMatrixOf(_target.get());
            osg::Vec3d targetPos = world.getTrans();

            osg::Vec3d center = targetPos;
            osg::Vec3d eye = targetPos + osg::Vec3d(0.0, 0.0, 0.0);

            if (_alignYaw)
            {
                osg::Quat rot = world.getRotate();
                eye = targetPos + rot * _offset;
            }
            else
            {
                eye = targetPos + _offset;
            }

            setCenter(center);
            setHomePosition(eye, center, osg::Vec3d(0, 0, 1));
        }
        return handled;
    }

private:
    osg::observer_ptr<osg::Node> _target;
    osg::Vec3d _offset;
    bool _alignYaw;
};

// --- Main ---
int main(int argc, char** argv)
{
    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::Node* model = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");
    if (!model) return 1;

    osg::ref_ptr<osg::MatrixTransform> trans = new osg::MatrixTransform;
    trans->addUpdateCallback(osgCookBook::createAnimationPathCallback(100.0f, 20.0f));
    trans->addChild(model);

    osg::ref_ptr<osg::MatrixTransform> terrain = new osg::MatrixTransform;
    terrain->addChild(osgDB::readNodeFile(dataPath + "lz.osg"));
    terrain->setMatrix(osg::Matrix::translate(0.0f, 0.0f, -200.0f));

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(trans.get());
    root->addChild(terrain.get());

    // --- Create manipulators ---
    osg::ref_ptr<osgGA::OrbitManipulator> orbit = new osgGA::OrbitManipulator;

    osg::ref_ptr<osgGA::NodeTrackerManipulator> nodeTracker = new osgGA::NodeTrackerManipulator;
    nodeTracker->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);
    nodeTracker->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);
    nodeTracker->setTrackNode(trans.get());

    osg::ref_ptr<FollowOrbitManipulator> follow = new FollowOrbitManipulator(trans.get());
    follow->setOffset(osg::Vec3d(0.0, -80.0, 25.0));  // farther & higher chase
    follow->setAlignYaw(true);

    osg::Vec3d center0 = osg::Vec3d(0, 0, 0);
    osg::Vec3d eye0    = center0 + osg::Vec3d(0.0, -60.0, 25.0);
    osg::Vec3d up0(0, 0, 1);
    orbit->setHomePosition(eye0, center0, up0);
    nodeTracker->setHomePosition(eye0, center0, up0);
    follow->setHomePosition(eye0, center0, up0);

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keySwitch = new osgGA::KeySwitchMatrixManipulator;
    keySwitch->addMatrixManipulator('1', "Orbit", orbit.get());
    keySwitch->addMatrixManipulator('2', "NodeTracker", nodeTracker.get());
    keySwitch->addMatrixManipulator('3', "FollowOrbit", follow.get());

    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 800, 600);
    viewer.setSceneData(root.get());
    viewer.setCameraManipulator(keySwitch.get());

    std::cout << "Press 1: OrbitManipulator\n";
    std::cout << "Press 2: NodeTrackerManipulator\n";
    std::cout << "Press 3: FollowOrbitManipulator (chase-like follow)\n";

    return viewer.run();
}