#include <osg/Group>
#include <osg/Geode>
#include <osgText/Text>
#include <osg/PositionAttitudeTransform>
#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
#include <osg/MatrixTransform>
#include <osg/Timer>
#include <cmath>
#include <sstream>

// ========================= Animation Callback =========================
class CessnaUpdateCallback : public osg::NodeCallback
{
public:
    CessnaUpdateCallback() : _startTime(osg::Timer::instance()->time_s()) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        osg::PositionAttitudeTransform* pat = dynamic_cast<osg::PositionAttitudeTransform*>(node);
        if (!pat) return;

        double t = osg::Timer::instance()->time_s() - _startTime;

        // Circular motion parameters
        float radius = 100.0f;
        float angularSpeed = 0.5f; // radians/sec
        float height = 30.0f;

        // Circular path
        float x = radius * std::cos(angularSpeed * t);
        float y = radius * std::sin(angularSpeed * t);
        float z = height;

        // Yaw rotation along tangent direction
        float yaw = angularSpeed * t + osg::PI_2; // face forward
        osg::Quat rotation;
        rotation.makeRotate(yaw, osg::Vec3(0, 0, 1));

        // Apply transform
        pat->setPosition(osg::Vec3(x, y, z));
        pat->setAttitude(rotation);

        // Continue traversing
        traverse(node, nv);
    }

private:
    double _startTime;
};

// ========================= Main =========================
int main(int argc, char** argv)
{
    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 1280, 720);

    osg::ref_ptr<osg::Group> root = new osg::Group();

    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    // Load Cessna model
    osg::ref_ptr<osg::Node> cessna = osgDB::readNodeFile(dataPath + "cessna.osg");
    if (!cessna)
    {
        osg::notify(osg::FATAL) << "Cannot load cessna.osg!" << std::endl;
        return 1;
    }

    // --- Create main transform (motion + rotation) ---
    osg::ref_ptr<osg::PositionAttitudeTransform> cessnaXform = new osg::PositionAttitudeTransform();
    cessnaXform->addChild(cessna);

    // --- Create a text node attached to Cessna (local offset) ---
    osg::ref_ptr<osgText::Text> text = new osgText::Text();
    text->setFont("fonts/arial.ttf");
    text->setCharacterSize(10.0f);
    text->setAxisAlignment(osgText::TextBase::SCREEN);
    text->setColor(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));
    text->setText("Right Wing Label");

    osg::ref_ptr<osg::Geode> textGeode = new osg::Geode();
    textGeode->addDrawable(text);

    // Offset: +Y = right wing, +Z = a bit above
    osg::ref_ptr<osg::PositionAttitudeTransform> textOffset = new osg::PositionAttitudeTransform();
    textOffset->setPosition(osg::Vec3(0.0f, 15.0f, 5.0f));
    textOffset->addChild(textGeode);

    // Attach to same Cessna transform → inherits motion and rotation
    cessnaXform->addChild(textOffset);

    // Add to scene
    root->addChild(cessnaXform);

    // Add callback for circular motion
    cessnaXform->setUpdateCallback(new CessnaUpdateCallback());

    viewer.setSceneData(root);
    viewer.realize();
    return viewer.run();
}
