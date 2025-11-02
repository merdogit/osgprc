#include <osg/Group>
#include <osg/Geode>
#include <osgText/Text>
#include <osg/PositionAttitudeTransform>
#include <osgViewer/Viewer>
#include <osgDB/ReadFile>

int main(int argc, char** argv)
{
    // Viewer setup
    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow(100, 100, 1280, 720);

    // Root node
    osg::ref_ptr<osg::Group> root = new osg::Group();

    const std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    // Load the Cessna model (use OSG's built-in model)
    osg::ref_ptr<osg::Node> cessna = osgDB::readNodeFile(dataPath + "cessna.osg");
    if (!cessna)
    {
        osg::notify(osg::FATAL) << "Cannot find cessna.osg model!" << std::endl;
        return 1;
    }

    // Attach it to a transform so we can position and scale it
    osg::ref_ptr<osg::PositionAttitudeTransform> cessnaXform = new osg::PositionAttitudeTransform();
    cessnaXform->setScale(osg::Vec3(1.0f, 1.0f, 1.0f));
    cessnaXform->setPosition(osg::Vec3(0.0f, 0.0f, 0.0f));
    cessnaXform->addChild(cessna);

    // Create a text label above the Cessna
    osg::ref_ptr<osgText::Text> text = new osgText::Text();
    text->setFont("fonts/arial.ttf"); // You can use another font if needed
    text->setCharacterSize(4.0f);
    text->setAxisAlignment(osgText::TextBase::SCREEN); // Always face the screen
    text->setPosition(osg::Vec3(0.0f, 0.0f, 20.0f));   // Position above the plane
    text->setText("Cessna 172 - Alt: 1200m  Spd: 210 km/h");
    text->setColor(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text

    // Place text inside a geode and attach to the same transform
    osg::ref_ptr<osg::Geode> textGeode = new osg::Geode();
    textGeode->addDrawable(text);
    cessnaXform->addChild(textGeode);

    // Add to scene
    root->addChild(cessnaXform);

    // Set up viewer
    viewer.setSceneData(root);
    viewer.realize();

    return viewer.run();
}
