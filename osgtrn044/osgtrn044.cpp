#include <osg/Group>
#include <osg/Geode>
#include <osgText/Text>
#include <osg/PositionAttitudeTransform>
#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
#include <osg/Timer>
#include <osgViewer/Viewer>
#include <cmath>

// ========================= Text Size Update Callback =========================
class TextScaleCallback : public osg::NodeCallback
{
public:
    TextScaleCallback(osgText::Text* text) : _text(text) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        if (!_text) return;

        // Get eye (camera) position
        osg::Vec3d eye = nv->getEyePoint();

        // Get text world position (center of bounding box)
        osg::Vec3d textPos = _text->getBound().center();

        // Compute distance
        double distance = (eye - textPos).length();

        // Adjust font size with distance
        float baseSize = 10.0f;
        float scaleFactor = 0.03f; // smaller value = less scaling
        float newSize = baseSize + distance * scaleFactor;

        _text->setCharacterSize(newSize);

        traverse(node, nv);
    }

private:
    osg::observer_ptr<osgText::Text> _text;
};

// ========================= Cessna Circular Motion =========================
class CessnaUpdateCallback : public osg::NodeCallback
{
public:
    CessnaUpdateCallback() : _startTime(osg::Timer::instance()->time_s()) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        osg::PositionAttitudeTransform* pat = dynamic_cast<osg::PositionAttitudeTransform*>(node);
        if (!pat) return;

        double t = osg::Timer::instance()->time_s() - _startTime;

        float radius = 100.0f;
        float angularSpeed = 0.5f;
        float height = 30.0f;

        float x = radius * std::cos(angularSpeed * t);
        float y = radius * std::sin(angularSpeed * t);
        float z = height;

        float yaw = angularSpeed * t + osg::PI_2;
        osg::Quat rotation;
        rotation.makeRotate(yaw, osg::Vec3(0, 0, 1));

        pat->setPosition(osg::Vec3(x, y, z));
        pat->setAttitude(rotation);

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

    // --- Cessna transform ---
    osg::ref_ptr<osg::PositionAttitudeTransform> cessnaXform = new osg::PositionAttitudeTransform();
    cessnaXform->addChild(cessna);

    // --- Text ---
    osg::ref_ptr<osgText::Text> text = new osgText::Text();
    text->setFont("fonts/arial.ttf");
    text->setCharacterSize(10.0f);
    text->setAxisAlignment(osgText::TextBase::SCREEN);
    text->setColor(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));
    text->setText("Right Wing Label");

    osg::ref_ptr<osg::Geode> textGeode = new osg::Geode();
    textGeode->addDrawable(text);

    osg::ref_ptr<osg::PositionAttitudeTransform> textOffset = new osg::PositionAttitudeTransform();
    textOffset->setPosition(osg::Vec3(0.0f, 15.0f, 5.0f)); // offset: right wing + above
    textOffset->addChild(textGeode);

    // Font scaling callback
    textGeode->setUpdateCallback(new TextScaleCallback(text));

    // Add to plane
    cessnaXform->addChild(textOffset);

    // Motion callback
    cessnaXform->setUpdateCallback(new CessnaUpdateCallback());

    root->addChild(cessnaXform);
    viewer.setSceneData(root);
    viewer.setUpViewInWindow(100, 100, 600, 600);

    viewer.realize();
    return viewer.run();
}