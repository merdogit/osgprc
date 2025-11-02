#include <osg/AnimationPath>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/NodeTrackerManipulator>
#include <osg/Geode>
#include <osg/Geometry>

osg::AnimationPath *createAnimationPath(float radius, float time)
{
    osg::ref_ptr<osg::AnimationPath> path = new osg::AnimationPath;
    path->setLoopMode(osg::AnimationPath::LOOP);

    unsigned int numSamples = 64;
    float delta_yaw = 2.0f * osg::PI / ((float)numSamples - 1.0f);
    float delta_time = time / (float)numSamples;

    for (unsigned int i = 0; i < numSamples; ++i)
    {
        float yaw = delta_yaw * (float)i;
        osg::Vec3 pos(sinf(yaw) * radius, cosf(yaw) * radius, 0.0f);
        osg::Quat rot(-yaw, osg::Z_AXIS);
        path->insert(delta_time * (float)i, osg::AnimationPath::ControlPoint(pos, rot));
    }

    return path.release();
}

osg::Node* createReferenceCircle(float radius, unsigned int segments = 128)
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;

    // Build circle vertices
    for (unsigned int i = 0; i <= segments; ++i)
    {
        float angle = 2.0f * osg::PI * ((float)i / (float)segments);
        vertices->push_back(osg::Vec3(sinf(angle) * radius, cosf(angle) * radius, 0.0f));
    }

    // Single color (red)
    colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    geom->setVertexArray(vertices.get());
    geom->setColorArray(colors.get(), osg::Array::BIND_OVERALL);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP_ADJACENCY, 0, vertices->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(geom.get());

    return geode.release();
}

int main()
{
    std::string dataPath = "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/";

    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(dataPath + "cessna.osg.0,0,90.rot");

    // Plane transform (animated)
    osg::ref_ptr<osg::MatrixTransform> planeXform = new osg::MatrixTransform;
    planeXform->addChild(model.get());

    osg::ref_ptr<osg::AnimationPathCallback> apcb = new osg::AnimationPathCallback;
    float radius = 100.0f;
    apcb->setAnimationPath(createAnimationPath(radius, 10.0f));
    planeXform->setUpdateCallback(apcb.get());

    // --- Tail offset node ---
    osg::ref_ptr<osg::MatrixTransform> tailNode = new osg::MatrixTransform;
    tailNode->setMatrix(osg::Matrix::translate(0.0f, -15.0f, 3.0f));
    planeXform->addChild(tailNode.get());

    // --- Reference circle ---
    osg::ref_ptr<osg::Node> referenceCircle = createReferenceCircle(radius);

    // Root node
    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(planeXform.get());
    root->addChild(referenceCircle.get());

    osgViewer::Viewer viewer;

    // NodeTrackerManipulator tracks the tail node
    osg::ref_ptr<osgGA::NodeTrackerManipulator> manip = new osgGA::NodeTrackerManipulator;
    manip->setTrackNode(tailNode.get());
    manip->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);

    manip->setHomePosition(
        osg::Vec3(-10.0, -5.0, 2.0),
        osg::Vec3(0.0, 0.0, 0.0),
        osg::Vec3(0.0, 0.0, 1.0)
    );

    viewer.setCameraManipulator(manip.get());
    viewer.setSceneData(root.get());
    viewer.setUpViewInWindow(700, 50, 600, 600);

    return viewer.run();
}