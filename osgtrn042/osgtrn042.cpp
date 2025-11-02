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
    CessnaUpdateCallback(osgText::Text* text)
        : _text(text), _startTime(osg::Timer::instance()->time_s()) {}

    void operator()(osg::Node* node, osg::NodeVisitor* nv) override
    {
        osg::PositionAttitudeTransform* pat = dynamic_cast<osg::PositionAttitudeTransform*>(node);
        if (!pat) return;

        double t = osg::Timer::instance()->time_s() - _startTime;

        // Circular motion parameters
        float radius = 100.0f;
        float speed = 0.5f; // radians per second
        float height = 30.0f;

        // Compute circular position
        float x = radius * std::cos(speed * t);
        float y = radius * std::sin(speed * t);
        float z = height;

        // Update position
        pat->setPosition(osg::Vec3(x, y, z));

        // Face tangent direction
        float yaw = speed * t + osg::PI_2; // nose points along tangent
        osg::Quat rot;
        rot.makeRotate(yaw, osg::Vec3(0, 0, 1));
        pat->setAttitude(rot);

        // Update text position (slightly above the plane)
        if (_text.valid())
        {
            _text->setPosition(osg::Vec3(x, y + 20.0f, z + 20.0f));

            std::ostringstream ss;
            ss.setf(std::ios::fixed);
            ss.precision(2);
            ss << "Cessna\n"
               << "X: " << x << "\n"
               << "Y: " << y << "\n"
               << "Z: " << z;
            _text->setText(ss.str());
        }

        // Continue traversing
        traverse(node, nv);
    }

private:
    osg::observer_ptr<osgText::Text> _text;
    double _startTime;
};

// ========================= Main =========================
int main(int argc, char** argv)
{
    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 1280, 720);

    osg::ref_ptr<osg::Group> root = new osg::Group();

    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    // Load model
    osg::ref_ptr<osg::Node> cessna = osgDB::readNodeFile(dataPath + "cessna.osg");
    if (!cessna)
    {
        osg::notify(osg::FATAL) << "Cannot load cessna.osg!" << std::endl;
        return 1;
    }

    // Transform for motion
    osg::ref_ptr<osg::PositionAttitudeTransform> cessnaXform = new osg::PositionAttitudeTransform();
    cessnaXform->addChild(cessna);

    // Text
    osg::ref_ptr<osgText::Text> text = new osgText::Text();
    text->setFont("fonts/arial.ttf");
    text->setCharacterSize(10.0f);
    text->setAxisAlignment(osgText::TextBase::SCREEN);
    text->setColor(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));

    osg::ref_ptr<osg::Geode> textGeode = new osg::Geode();
    textGeode->addDrawable(text);
    root->addChild(textGeode);

    // Add callback for motion and text update
    cessnaXform->setUpdateCallback(new CessnaUpdateCallback(text));

    root->addChild(cessnaXform);
    viewer.setSceneData(root);
    viewer.realize();

    return viewer.run();
}
