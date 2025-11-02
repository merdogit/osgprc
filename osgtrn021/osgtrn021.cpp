#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/MatrixTransform>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osg/NodeCallback>
#include <osg/StateSet>
#include <osg/Image>

// ---------- Create a simple color texture ----------
osg::ref_ptr<osg::Image> createColorImage(int size = 256, osg::Vec4 color = osg::Vec4(0.6, 0.6, 0.6, 1.0))
{
    osg::ref_ptr<osg::Image> image = new osg::Image();
    image->allocateImage(size, size, 1, GL_RGBA, GL_UNSIGNED_BYTE);

    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            unsigned char* pixel = image->data(x, y);
            pixel[0] = static_cast<unsigned char>(color.r() * 255);
            pixel[1] = static_cast<unsigned char>(color.g() * 255);
            pixel[2] = static_cast<unsigned char>(color.b() * 255);
            pixel[3] = static_cast<unsigned char>(color.a() * 255);
        }
    }
    return image;
}

// ---------- Create the textured ground ----------
osg::ref_ptr<osg::Geode> createGround(float size = 1000.0f)
{
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    vertices->push_back(osg::Vec3(-size, 0, -size));
    vertices->push_back(osg::Vec3(size, 0, -size));
    vertices->push_back(osg::Vec3(size, 0, size));
    vertices->push_back(osg::Vec3(-size, 0, size));
    geom->setVertexArray(vertices);

    osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array();
    texcoords->push_back(osg::Vec2(0.0f, 0.0f));
    texcoords->push_back(osg::Vec2(10.0f, 0.0f));
    texcoords->push_back(osg::Vec2(10.0f, 10.0f));
    texcoords->push_back(osg::Vec2(0.0f, 10.0f));
    geom->setTexCoordArray(0, texcoords);

    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(GL_QUADS);
    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(3);
    geom->addPrimitiveSet(indices);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();
    texture->setImage(createColorImage());
    texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
    texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);

    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

    return geode;
}

// ---------- Callback to move ground under camera ----------
class GroundFollowCallback : public osg::NodeCallback
{
public:
    GroundFollowCallback(osg::MatrixTransform* ground, osg::Camera* cam)
        : _ground(ground), _camera(cam) {}

    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        if (!_camera.valid() || !_ground.valid()) return;

        osg::Vec3 eye, center, up;
        _camera->getViewMatrixAsLookAt(eye, center, up);

        osg::Matrix mat = osg::Matrix::translate(osg::Vec3(eye.x(), 0.0, eye.z()));
        _ground->setMatrix(mat);

        traverse(node, nv);
    }

private:
    osg::ref_ptr<osg::MatrixTransform> _ground;
    osg::observer_ptr<osg::Camera> _camera;
};

// ---------- Main ----------
int main()
{
    osgViewer::Viewer viewer;

    // Create ground
    osg::ref_ptr<osg::Geode> groundGeode = createGround(5000.0f);
    osg::ref_ptr<osg::MatrixTransform> groundTransform = new osg::MatrixTransform();
    groundTransform->addChild(groundGeode);

    // Set camera manipulator
    osg::ref_ptr<osgGA::TrackballManipulator> manip = new osgGA::TrackballManipulator();
    viewer.setCameraManipulator(manip);

    // Attach callback to move ground under camera
    groundTransform->setUpdateCallback(new GroundFollowCallback(groundTransform, viewer.getCamera()));

    // Scene root
    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->addChild(groundTransform);

    viewer.setSceneData(root);
    viewer.realize();

    return viewer.run();
}