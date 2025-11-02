#include <osg/Geometry>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/GUIEventHandler>
#include <osg/AnimationPath>
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
        if (!_activeNode) return false;

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
            case osgGA::GUIEventAdapter::KEY_Left:   trans.x() -= step; break;
            case osgGA::GUIEventAdapter::KEY_Right:  trans.x() += step; break;
            case osgGA::GUIEventAdapter::KEY_Up:     trans.y() += step; break;
            case osgGA::GUIEventAdapter::KEY_Down:   trans.y() -= step; break;
            case osgGA::GUIEventAdapter::KEY_Page_Up:   trans.z() += step; break;
            case osgGA::GUIEventAdapter::KEY_Page_Down: trans.z() -= step; break;

            // Rotation
            case 'q': rot = osg::Quat(angleStep, osg::Vec3(0,0,1)) * rot; break; // yaw left
            case 'e': rot = osg::Quat(-angleStep, osg::Vec3(0,0,1)) * rot; break; // yaw right
            case 'w': rot = osg::Quat(angleStep, osg::Vec3(1,0,0)) * rot; break; // pitch up
            case 's': rot = osg::Quat(-angleStep, osg::Vec3(1,0,0)) * rot; break; // pitch down
            case 'a': rot = osg::Quat(angleStep, osg::Vec3(0,1,0)) * rot; break; // roll left
            case 'd': rot = osg::Quat(-angleStep, osg::Vec3(0,1,0)) * rot; break; // roll right
            default: return false;
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
            case '1': _manip->setTrackNode(_tail.get()); _manip->home(0.0); _adjustHandler->setActiveNode(_tail.get()); std::cout<<"Switched to TAIL camera\n"; return true;
            case '2': _manip->setTrackNode(_wing.get()); _manip->home(0.0); _adjustHandler->setActiveNode(_wing.get()); std::cout<<"Switched to WING camera\n"; return true;
            case '3': _manip->setTrackNode(_cockpit.get()); _manip->home(0.0); _adjustHandler->setActiveNode(_cockpit.get()); std::cout<<"Switched to COCKPIT camera\n"; return true;
            }
        }
        return false;
    }

private:
    osg::observer_ptr<osgGA::NodeTrackerManipulator> _manip;
    osg::observer_ptr<osg::MatrixTransform> _tail, _wing, _cockpit;
    osg::observer_ptr<CameraAdjustHandler> _adjustHandler;
};

// ===== Animation path with yaw and roll =====
osg::AnimationPath* createAnimationPath(float radius, float time)
{
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::LOOP);

    unsigned int numSamples = 64;
    float delta_yaw = 2.0f * osg::PI / ((float)numSamples - 1.0f);
    float delta_time = time / (float)numSamples;

    for (unsigned int i = 0; i < numSamples; ++i)
    {
        float yaw = delta_yaw * (float)i;
        osg::Vec3 pos(sinf(yaw)*radius, cosf(yaw)*radius, 0.0f);

        // Add dynamic roll & yaw
        float rollAmplitude = osg::DegreesToRadians(1.0f);
        float roll = sinf(yaw*4.0f) * rollAmplitude;
        float extraYaw = sinf(yaw*2.0f) * osg::DegreesToRadians(5.0f);

        osg::Quat rot(-yaw + extraYaw, osg::Z_AXIS); // yaw
        rot *= osg::Quat(roll, osg::Y_AXIS);         // roll along forward axis

        path->insert(delta_time*(float)i, osg::AnimationPath::ControlPoint(pos, rot));
    }

    return path.release();
}

// ===== Reference circle =====
osg::Node* createReferenceCircle(float radius, unsigned int segments = 128)
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;

    for (unsigned int i = 0; i <= segments; ++i)
    {
        float angle = 2.0f * osg::PI * ((float)i / (float)segments);
        vertices->push_back(osg::Vec3(sinf(angle)*radius, cosf(angle)*radius, 0.0f));
    }
    colors->push_back(osg::Vec4(1.0f,0.0f,0.0f,1.0f));

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    geom->setVertexArray(vertices.get());
    geom->setColorArray(colors.get(), osg::Array::BIND_OVERALL);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP_ADJACENCY, 0, vertices->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(geom.get());

    return geode.release();
}

// ===== Main =====
int main()
{
    osg::ref_ptr<osg::Group> root = new osg::Group();

    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";
    osg::ref_ptr<osg::Node> cessna = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");
    if (!cessna) { std::cerr << "Error: Could not load cessna.osg\n"; return 1; }

    osg::ref_ptr<osg::MatrixTransform> cessnaXform = new osg::MatrixTransform;
    cessnaXform->addChild(cessna.get());
    root->addChild(cessnaXform.get());

    // Animation path
    osg::ref_ptr<osg::AnimationPathCallback> apcb = new osg::AnimationPathCallback;
    float radius = 100.0f;
    apcb->setAnimationPath(createAnimationPath(radius, 10.0f));
    cessnaXform->setUpdateCallback(apcb.get());

    // Reference circle
    osg::ref_ptr<osg::Node> referenceCircle = createReferenceCircle(radius);
    root->addChild(referenceCircle.get());

    // Camera markers
    osg::ref_ptr<osg::MatrixTransform> tailNode = new osg::MatrixTransform;
    osg::ref_ptr<osg::MatrixTransform> wingNode = new osg::MatrixTransform;
    osg::ref_ptr<osg::MatrixTransform> cockpitNode = new osg::MatrixTransform;

    tailNode->setMatrix(osg::Matrix::rotate(
        osg::Quat(-0.0984102,0.0984102,-0.700225,0.700225)) * osg::Matrix::translate(-8.0f,1.0f,3.0f));
    wingNode->setMatrix(osg::Matrix::rotate(
        osg::Quat(0.0,0.022433,-0.642396,0.766044)) * osg::Matrix::translate(65.5f,2.5f,2.0f));
    cockpitNode->setMatrix(osg::Matrix::rotate(
        osg::Quat(-0.0500815,0.0523467,-0.715505,0.694841)) * osg::Matrix::translate(63.5f,-2.0f,2.0f));

    cessnaXform->addChild(tailNode.get());
    cessnaXform->addChild(wingNode.get());
    cessnaXform->addChild(cockpitNode.get());

    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(700,50,600,600);
    viewer.setSceneData(root.get());

    osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
    manip->setTrackNode(tailNode.get());
    manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_AZIM);
    manip->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);
    viewer.setCameraManipulator(manip.get());

    osg::ref_ptr<CameraAdjustHandler> adjustHandler = new CameraAdjustHandler(tailNode.get());
    viewer.addEventHandler(adjustHandler.get());

    osg::ref_ptr<CameraSwitchHandler> switchHandler =
        new CameraSwitchHandler(manip.get(), tailNode.get(), wingNode.get(), cockpitNode.get(), adjustHandler.get());
    viewer.addEventHandler(switchHandler.get());

    return viewer.run();
}
