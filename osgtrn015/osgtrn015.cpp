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

// ===== Keyboard Handler to adjust active camera marker =====
class CameraAdjustHandler : public osgGA::GUIEventHandler
{
public:
    CameraAdjustHandler(osg::MatrixTransform *activeNode)
        : _activeNode(activeNode) {}

    void setActiveNode(osg::MatrixTransform *node) { _activeNode = node; }

    bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &) override
    {
        if (!_activeNode)
            return false;

        if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
        {
            osg::Matrix m = _activeNode->getMatrix();
            osg::Vec3 trans = m.getTrans();
            osg::Quat rot = m.getRotate();
            float step = 0.5f;
            float angleStep = osg::DegreesToRadians(2.0f);

            switch (ea.getKey())
            {
            // Translation
            case osgGA::GUIEventAdapter::KEY_Left:
                trans.x() -= step;
                break;
            case osgGA::GUIEventAdapter::KEY_Right:
                trans.x() += step;
                break;
            case osgGA::GUIEventAdapter::KEY_Up:
                trans.y() += step;
                break;
            case osgGA::GUIEventAdapter::KEY_Down:
                trans.y() -= step;
                break;
            case osgGA::GUIEventAdapter::KEY_Page_Up:
                trans.z() += step;
                break;
            case osgGA::GUIEventAdapter::KEY_Page_Down:
                trans.z() -= step;
                break;

            // Rotation
            case 'q':
                rot = osg::Quat(angleStep, osg::Vec3(0, 0, 1)) * rot;
                break; // yaw left
            case 'e':
                rot = osg::Quat(-angleStep, osg::Vec3(0, 0, 1)) * rot;
                break; // yaw right
            case 'w':
                rot = osg::Quat(angleStep, osg::Vec3(1, 0, 0)) * rot;
                break; // pitch up
            case 's':
                rot = osg::Quat(-angleStep, osg::Vec3(1, 0, 0)) * rot;
                break; // pitch down
            case 'a':
                rot = osg::Quat(angleStep, osg::Vec3(0, 1, 0)) * rot;
                break; // roll left
            case 'd':
                rot = osg::Quat(-angleStep, osg::Vec3(0, 1, 0)) * rot;
                break; // roll right
            case 'r':
                _activeNode->setMatrix(_activeNode->getInverseMatrix() *
                                       _activeNode->getMatrix());
                trans = _activeNode->getMatrix().getTrans();
                rot = _activeNode->getMatrix().getRotate();
                break;
            default:
                return false;
            }

            _activeNode->setMatrix(osg::Matrix::rotate(rot) * osg::Matrix::translate(trans));
            std::cout << "Marker position: "
                      << trans.x() << ", " << trans.y() << ", " << trans.z()
                      << " | rotation: " << rot.x() << ", " << rot.y() << ", " << rot.z() << ", " << rot.w()
                      << std::endl;
            return true;
        }
        return false;
    }

private:
    osg::observer_ptr<osg::MatrixTransform> _activeNode;
};

// ===== Keyboard Handler to switch camera modes =====
class CameraSwitchHandler : public osgGA::GUIEventHandler
{
public:
    CameraSwitchHandler(osgGA::NodeTrackerManipulator *manip,
                        osg::MatrixTransform *tail,
                        osg::MatrixTransform *wing,
                        osg::MatrixTransform *cockpit,
                        CameraAdjustHandler *adjustHandler)
        : _manip(manip), _tail(tail), _wing(wing), _cockpit(cockpit),
          _adjustHandler(adjustHandler) {}

    bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &) override
    {
        if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
        {
            switch (ea.getKey())
            {
            case '1': // Tail view
                _manip->setTrackNode(_tail.get());
                _manip->home(0.0);
                _adjustHandler->setActiveNode(_tail.get());
                std::cout << "Switched to TAIL camera\n";
                return true;
            case '2': // Wing view
                _manip->setTrackNode(_wing.get());
                _manip->home(0.0);
                _adjustHandler->setActiveNode(_wing.get());
                std::cout << "Switched to WING camera\n";
                return true;
            case '3': // Cockpit view
                _manip->setTrackNode(_cockpit.get());
                _manip->home(0.0);
                _adjustHandler->setActiveNode(_cockpit.get());
                std::cout << "Switched to COCKPIT camera\n";
                return true;
            }
        }
        return false;
    }

private:
    osg::observer_ptr<osgGA::NodeTrackerManipulator> _manip;
    osg::observer_ptr<osg::MatrixTransform> _tail, _wing, _cockpit;
    osg::observer_ptr<CameraAdjustHandler> _adjustHandler;
};

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

    // Camera marker nodes
    osg::ref_ptr<osg::MatrixTransform> tailNode = new osg::MatrixTransform;
    osg::ref_ptr<osg::MatrixTransform> wingNode = new osg::MatrixTransform;
    osg::ref_ptr<osg::MatrixTransform> cockpitNode = new osg::MatrixTransform;

    // Initial positions (adjust later with keyboard)
    tailNode->setMatrix(osg::Matrix::rotate(osg::Quat(-0.571967, 0.415509, 0.57267, 0.415036)) *
                        osg::Matrix::translate(0.5, 4, 0));
    wingNode->setMatrix(osg::Matrix::rotate(osg::Quat(-0.491198, 0.50865, 0.491198, 0.50865)) *
                        osg::Matrix::translate(-21.5, 1, -6));
    cockpitNode->setMatrix(osg::Matrix::rotate(osg::Quat(-0.464871, 0.534563, 0.464336, 0.531534)) *
                        osg::Matrix::translate(-31.5, 6, 0));

    modelXform->addChild(tailNode.get());
    modelXform->addChild(wingNode.get());
    modelXform->addChild(cockpitNode.get());

    modelXform->setMatrix(osg::Matrix::rotate(osg::Quat(-0.00612029, -0.700665, 0.713439, 0.00601263)));

    osgViewer::Viewer viewer;
    viewer.setSceneData(root.get());

    // Camera manipulator
    osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
    manip->setTrackNode(tailNode.get());
    manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);
    manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);
    viewer.setCameraManipulator(manip.get());

    // Handlers
    osg::ref_ptr<CameraAdjustHandler> adjustHandler = new CameraAdjustHandler(tailNode.get());
    viewer.addEventHandler(adjustHandler.get());

    osg::ref_ptr<CameraSwitchHandler> switchHandler =
        new CameraSwitchHandler(manip.get(), tailNode.get(), wingNode.get(), cockpitNode.get(), adjustHandler.get());
    viewer.addEventHandler(switchHandler.get());
    viewer.setUpViewInWindow(700, 50, 800, 600);

    return viewer.run();
}