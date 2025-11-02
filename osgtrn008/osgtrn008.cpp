#include <osg/Node>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <iostream>

// Search recursively for a node by name
osg::ref_ptr<osg::Node> findNodeByName(const std::string& name, osg::Node* root) {
    if (!root) return nullptr;

    if (root->getName() == name) {
        return root;
    }

    osg::Group* group = root->asGroup();
    if (group) {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i) {
            osg::ref_ptr<osg::Node> found = findNodeByName(name, group->getChild(i));
            if (found) return found;
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    std::string filename = (argc > 1) ? argv[1] : "/home/murate/Documents/SwTrn/OsgTrn/OpenSceneGraph-Data/F-14-low-poly-osg.ac";
    osg::ref_ptr<osg::Node> root = osgDB::readNodeFile(filename);
    if (!root) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 1;
    }

    // Find the UC-R landing gear
    osg::ref_ptr<osg::Node> gearNode = findNodeByName("UC-R", root);
    if (gearNode) {
        std::cout << "Found UC-R landing gear!" << std::endl;

        // Wrap it inside a MatrixTransform so we can animate it
        osg::ref_ptr<osg::MatrixTransform> gearTransform = new osg::MatrixTransform;
        osg::Group* parent = gearNode->getParent(0);
        if (parent) {
            parent->replaceChild(gearNode, gearTransform);  // replace original with transform
            gearTransform->addChild(gearNode);
        }

        // Example: translate the gear down (like lowering it)
        osg::Matrix m;
        m.makeTranslate(0.0f, 0.0f, -2.0f); // move down 2 units
        gearTransform->setMatrix(m);
    } else {
        std::cout << "UC-R not found!" << std::endl;
    }

    // Show the model
    osgViewer::Viewer viewer;
    viewer.setSceneData(root.get());
    return viewer.run();
}