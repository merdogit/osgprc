#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osgViewer/Viewer>

// Helper to create one axis (bar + cone + label)
osg::ref_ptr<osg::MatrixTransform> createAxis 
(
    const osg::Vec3& axisDir,
    const osg::Vec4& color,
    const std::string& label,
    float length = 5.0f,
    float radius = 0.1f,
    float coneRadius = 0.2f,
    float coneHeight = 0.5f)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;

    // Cylinder, starts from origin and goes along +Z
    osg::ref_ptr<osg::Cylinder> bar = new osg::Cylinder(osg::Vec3(0, 0, length * 0.5f), radius, length);
    osg::ref_ptr<osg::ShapeDrawable> barDrawable = new osg::ShapeDrawable(bar);
    barDrawable->setColor(color);
    geode->addDrawable(barDrawable);

    // Cone tip at end of bar
    osg::ref_ptr<osg::Cone> arrow = new osg::Cone(osg::Vec3(0, 0, length), coneRadius, coneHeight);
    osg::ref_ptr<osg::ShapeDrawable> arrowDrawable = new osg::ShapeDrawable(arrow);
    arrowDrawable->setColor(color);
    geode->addDrawable(arrowDrawable);

    // Text label at tip
    osg::ref_ptr<osgText::Text> text = new osgText::Text;
    text->setFont("arial.ttf");
    text->setCharacterSize(0.7f);
    text->setAxisAlignment(osgText::Text::SCREEN);
    text->setPosition(osg::Vec3(0, 0, length + coneHeight + 0.2f));
    text->setText(label);
    text->setColor(color);
    geode->addDrawable(text);

    // Rotate whole axis to target direction
    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
    osg::Quat rot;
    rot.makeRotate(osg::Vec3(0, 0, 1), axisDir); // rotate Z into axisDir
    mt->setMatrix(osg::Matrix::rotate(rot));
    mt->addChild(geode);

    return mt;
}

int main()
{
    osg::ref_ptr<osg::Group> root = new osg::Group;

    // Sphere at origin (radius doubled)
    osg::ref_ptr<osg::Geode> originGeode = new osg::Geode;
    osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.3f);
    osg::ref_ptr<osg::ShapeDrawable> sphereDrawable = new osg::ShapeDrawable(sphere);
    sphereDrawable->setColor(osg::Vec4(1, 1, 1, 1)); // white
    originGeode->addDrawable(sphereDrawable);
    root->addChild(originGeode);

    // Axes with labels
    root->addChild(createAxis(osg::Vec3(1, 0, 0), osg::Vec4(1, 0, 0, 1), "X")); // red
    root->addChild(createAxis(osg::Vec3(0, 1, 0), osg::Vec4(0, 1, 0, 1), "Y")); // green
    root->addChild(createAxis(osg::Vec3(0, 0, 1), osg::Vec4(0, 0, 1, 1), "Z")); // blue

    osgViewer::Viewer viewer;
    // viewer.getCamera()->setClearColor(osg::Vec4(0.2, 0.2, 0.2, 1));
    viewer.setSceneData(root.get());
    viewer.setUpViewInWindow(100, 100, 600, 600);
    return viewer.run();
}