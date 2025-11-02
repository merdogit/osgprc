#pragma once
#include <osg/Node>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/OrbitManipulator>
#include <osgViewer/Viewer>

// Custom follow manipulator from before
#include "FollowOrbitManipulator.hpp"

class CameraController
{
public:
    CameraController(osg::Node *target)
    {
        // --- Initialize manipulators ---
        _orbit = new osgGA::OrbitManipulator;
        _nodeTracker = new osgGA::NodeTrackerManipulator;
        _follow = new FollowOrbitManipulator(target);

        // NodeTracker setup
        _nodeTracker->setTrackerMode(osgGA::NodeTrackerManipulator::NODE_CENTER_AND_ROTATION);
        _nodeTracker->setRotationMode(osgGA::NodeTrackerManipulator::TRACKBALL);
        _nodeTracker->setTrackNode(target);

        // Shared default camera positions
        osg::Vec3d center0 = osg::Vec3d(0, 0, 0);
        osg::Vec3d eye0(0.0, -60.0, 25.0);
        osg::Vec3d up0(0, 0, 1);
        _orbit->setHomePosition(eye0, center0, up0);
        _nodeTracker->setHomePosition(eye0, center0, up0);
        _follow->setHomePosition(eye0, center0, up0);

        // --- KeySwitch manipulator for switching ---
        _keySwitch = new osgGA::KeySwitchMatrixManipulator;
        _keySwitch->addMatrixManipulator('1', "Orbit", _orbit.get());
        _keySwitch->addMatrixManipulator('2', "NodeTracker", _nodeTracker.get());
        _keySwitch->addMatrixManipulator('3', "FollowOrbit", _follow.get());
        _keySwitch->selectMatrixManipulator(0);
    }

    // Attach to viewer
    void attach(osgViewer::Viewer &viewer)
    {
        viewer.setCameraManipulator(_keySwitch.get());
    }

    // Switch manually from code or ImGui (0=Orbit, 1=NodeTracker, 2=Follow)
    void setActiveMode(int index)
    {
        _keySwitch->selectMatrixManipulator(index);
    }

    // Query which manipulator is active
    int getActiveMode() const
    {
        osgGA::CameraManipulator *active = _keySwitch->getCurrentMatrixManipulator();
        for (unsigned int i = 0; i < _keySwitch->getNumMatrixManipulators(); ++i)
        {
            if (_keySwitch->getMatrixManipulatorWithIndex(i) == active)
                return i;
        }
        return -1; // none found
    }

    // Access manipulators if needed
    osgGA::OrbitManipulator *getOrbit() const { return _orbit.get(); }
    osgGA::NodeTrackerManipulator *getNodeTracker() const { return _nodeTracker.get(); }
    FollowOrbitManipulator *getFollow() const { return _follow.get(); }
    osgGA::KeySwitchMatrixManipulator *getKeySwitch() const { return _keySwitch.get(); }

private:
    osg::ref_ptr<osgGA::OrbitManipulator> _orbit;
    osg::ref_ptr<osgGA::NodeTrackerManipulator> _nodeTracker;
    osg::ref_ptr<FollowOrbitManipulator> _follow;
    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> _keySwitch;
};