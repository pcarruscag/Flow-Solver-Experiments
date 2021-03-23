//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "fileManagementTests.h"

#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <stdlib.h>
#include "../src/geometryParameters.h"
#include "../src/meshParameters.h"
#include "../src/flowParameters.h"
#include "../src/adjointParameters.h"

namespace fileManagement{

void testSuite()
{
    std::cout << std::endl;
    _geometryParametersTester();
    std::cout << std::endl;
    _meshParametersTester();
    std::cout << std::endl;
    _flowParametersTester();
    std::cout << std::endl;
    _adjointParametersTester();
}

void _geometryParametersTester()
{
    std::cout << "### GEOMETRY PARAMETERS MANAGER TESTS ###" << std::endl;

    int errCode, nBld;
    std::vector<double> x,y;
    std::vector<std::vector<double> > xmat, ymat;
    std::string filePath = "../../lib/fileManagement/tests/testInputFile.txt";
    GeometryParamManager manager;
    GeometryParamManager::advancedParams advParams;

    errCode = manager.readFile(filePath);
    assert(errCode==0);

    manager.getNblades(nBld);
    assert(nBld==9);

    manager.getHub(x,y);
    assert(y.size()==6);
    assert(x[5]==50.0);

    manager.getShroud(x,y);
    assert(x.size()==5);
    assert(y[4]==25.0);

    manager.getLe(x);
    manager.getTe(y);
    assert(x.size()==4);
    assert(y[2]==0.83);

    manager.getTheta(ymat);
    assert(ymat.size()==4);
    assert(ymat[0].size()==3);
    assert(ymat[3][2]==13.0);

    manager.getThick(xmat,ymat);
    assert(ymat.size()==4);
    assert(xmat[1].size()==2);
    assert(xmat[0][1]==1.0);
    assert(ymat[3][0]==2.0);

    manager.getAdvanced(advParams);
    assert(advParams.downsampleFactor==10);
}

void _meshParametersTester()
{
    std::cout << "### MESH PARAMETERS MANAGER TESTS ###" << std::endl;

    int errCode;
    std::string filePath = "../../lib/fileManagement/tests/meshSettings.txt";
    MeshParamManager manager;
    MeshParamManager::edgeParams edgePar;
    MeshParamManager::layerParams layerPar;
    MeshParamManager::volumeParams volumePar;
    MeshParamManager::advancedParams advPar;

    errCode = manager.readFile(filePath);
    assert(errCode==0);
    assert(manager.getEdgeParams(edgePar));
    assert(manager.getLayerParams(layerPar));
    assert(manager.getVolParams(volumePar));
    assert(manager.getAdvParams(advPar));

    assert(edgePar.nCellBld==110);
    assert(layerPar.nearWallSize==0.0001);
    assert(layerPar.orthogCtrl[0].CP[3]==1.9);
    assert(layerPar.orthogCtrl[0].weight==1.0);
    assert(volumePar.nearWallSize==0.0001);
    assert(advPar.decayFactor==3.0);
    assert(advPar.tol==0.05);
    assert(advPar.linSolTol==1e-6);
    assert(advPar.disableOrthogFraction==0.15);
}

void _flowParametersTester()
{
    std::cout << "### FLOW PARAMETERS MANAGER TESTS ###" << std::endl;

    int errCode;
    std::string filePath = "../../lib/fileManagement/tests/flowSettings.txt";
    FlowParamManager manager;

    errCode = manager.readFile(filePath);
    assert(errCode == 0);

    assert(manager.m_inletConditions.variable == manager.VELOCITY);
    assert(manager.m_inletConditions.coordinate == manager.CYLINDRICAL);
    assert(manager.m_inletConditions.scalar == 10);
    assert(manager.m_inletConditions.components[1] == 1);
    assert(abs(manager.m_inletConditions.turbVal2-0.2f)<1e-6f);

    assert(manager.m_outletConditions.pressure == 200000);
    assert(manager.m_outletConditions.massFlowOption);

    assert(manager.m_domainConditions.rotatingHub);
    assert(manager.m_domainConditions.rotationalSpeed == 3000);

    assert(abs(manager.m_fluidProperties.mu-1e-5f)<1e-10f);

    assert(abs(manager.m_controlParams.relax1-0.7f)<1e-6f);
    assert(abs(manager.m_controlParams.bigNumber-1e6f)<1);
}

void _adjointParametersTester()
{
    std::cout << "### ADJOINT PARAMETERS MANAGER TESTS ###" << std::endl;
    int errCode;
    std::string filePath = "../../lib/fileManagement/tests/adjointSettings.txt";
    AdjointParamManager manager;

    errCode = manager.readFile(filePath);
    assert(errCode == 0);

    std::vector<int> objectives;
    manager.getObjectives(objectives);
    assert(objectives[1]==8);

    int maxIters; double tolerance;
    manager.getStopCriteria(maxIters,tolerance);
    assert(maxIters==25 && tolerance==0.01);

    double amin, amax;
    manager.getRelaxLimits(amin,amax);
    assert(amin==0.05 && amax==1.5);

    std::string directory;
    manager.getScratchDir(directory);
    assert(directory.compare("test/mydir")==0);

    double minTol, ratioTol;
    manager.getLinSolParam(maxIters,tolerance,minTol,ratioTol);
    assert(maxIters==300 && tolerance==0.01 && minTol==1e-5 && ratioTol==1.0);

    bool firstOrder;
    manager.getFirstOrder(firstOrder);
    assert(!firstOrder);
}

} // namespace fileManagement
