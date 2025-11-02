#pragma once
#include <osgGA/OrbitManipulator>
#include <osg/Node>
#include <osg/Matrix>
#include <osg/ComputeBoundsVisitor>
#include <osg/Quat>
#include <osg/observer_ptr>
#include <iostream>

//
// FollowOrbitManipulator
// -----------------------
// A small extension of OrbitManipulator that automatically
// re-centers its orbit point to follow a moving target node.
//
class FollowOrbitManipulator : public osgGA::OrbitManipulator
{
public:
    explicit FollowOrbitManipulator(osg::Node* target = nullptr)
        : _target(target),
          _offset(0.0, -80.0, 25.0),
          _alignYaw(true)
    {
    }

    void setTarget(osg::Node* node) { _target = node; }
    void setOffset(const osg::Vec3d& off) { _offset = off; }
    void setAlignYaw(bool enable) { _alignYaw = enable; }

    // Main update hook
    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override
    {
        // Let the base manipulator process input first
        bool handled = osgGA::OrbitManipulator::handle(ea, aa);

        if (_target.valid() && ea.getEventType() == osgGA::GUIEventAdapter::FRAME)
        {
            const auto& paths = _target->getParentalNodePaths();
            if (!paths.empty())
            {
                osg::Matrix world = osg::computeLocalToWorld(paths.front());
                osg::Vec3d targetPos = world.getTrans();
                osg::Quat rotation = world.getRotate();

                // Compute eye and center positions
                osg::Vec3d center = targetPos;
                osg::Vec3d eye = _alignYaw ? targetPos + rotation * _offset
                                           : targetPos + _offset;

                setCenter(center);
                setHomePosition(eye, center, osg::Vec3d(0, 0, 1));
            }
        }

        return handled;
    }

private:
    osg::observer_ptr<osg::Node> _target;
    osg::Vec3d _offset;
    bool _alignYaw;
};
