//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flowParameters.h"

#include <iostream>

namespace fileManagement
{
FlowParamManager::FlowParamManager()
{
    m_dataIsReady = false;

    m_controlParams.turbulenceType = flow::turbulenceModels::ModelBase::SST;

    m_controlParams.tol = 1e-3;
    m_controlParams.cflNumber = 50.0;
    m_controlParams.initViscRatio = 100.0;
    m_controlParams.minIt = 30;
    m_controlParams.relax0 = 0.4;
    m_controlParams.relax0_t = 0.1;
    m_controlParams.maxIt = 100;
    m_controlParams.relax1 = 0.7;
    m_controlParams.relax1_t = 0.5;

    m_controlParams.mapTol = 1e-3;
    m_controlParams.venkatK = 0.027;
    m_controlParams.mdotRelax = 0.25;
    m_controlParams.linSolMaxIt = 100;
    m_controlParams.linSolTol = 1e-8;
    m_controlParams.bigNumber = 1.0e6;
}

int FlowParamManager::readFile(const string& filePath)
{
    ifstream inputFile;
    string line;
    stringstream strStream;

    m_dataIsReady = false;

    inputFile.open(filePath.c_str());
    if (inputFile.is_open())
    {
        getline(inputFile,line);
        if (line.compare(m_inletHeader)==0)
        {
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_inletConditions.variable;
            strStream >> m_inletConditions.direction;
            if(m_inletConditions.direction != NORMAL)
                strStream >> m_inletConditions.coordinate;
            m_getLineFromFile(inputFile,strStream);
            if(m_inletConditions.direction != COMPONENTS)
                strStream >> m_inletConditions.scalar;
            if(m_inletConditions.direction != NORMAL) {
                strStream >> m_inletConditions.components[0];
                strStream >> m_inletConditions.components[1];
                strStream >> m_inletConditions.components[2];
            }
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_inletConditions.turbSpec;
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_inletConditions.turbVal1;
            strStream >> m_inletConditions.turbVal2;
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_outletHeader)==0)
        {
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_outletConditions.pressure;
            strStream >> m_outletConditions.massFlowOption;
            if(m_outletConditions.massFlowOption)
                strStream >> m_outletConditions.massFlow;
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_domainHeader)==0)
        {
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_domainConditions.rotationalSpeed;
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_domainConditions.rotatingShroud;
            strStream >> m_domainConditions.rotatingHub;
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_fluidPropHeader)==0)
        {
            m_getLineFromFile(inputFile,strStream);
            strStream >> m_fluidProperties.rho;
            strStream >> m_fluidProperties.mu;
        } else
        {
            inputFile.close();
            return 2;
        }
        while(!inputFile.eof())
        {
            getline(inputFile,line);
            if (line.compare(m_fluidModelHeader)==0)
            {
                getline(inputFile,line);
                if (line.compare("Laminar")==0)
                    m_controlParams.turbulenceType = flow::turbulenceModels::ModelBase::LAMINAR;
                else if (line.compare("SST")==0)
                    m_controlParams.turbulenceType = flow::turbulenceModels::ModelBase::SST;
                else {
                    inputFile.close();
                    return 2;
                }
            } else if (line.compare(m_convergenceHeader)==0)
            {
                m_getLineFromFile(inputFile,strStream);
                strStream >> m_controlParams.tol;
                strStream >> m_controlParams.cflNumber;
                strStream >> m_controlParams.initViscRatio;
                m_getLineFromFile(inputFile,strStream);
                strStream >> m_controlParams.minIt;
                strStream >> m_controlParams.relax0;
                strStream >> m_controlParams.relax0_t;
                m_getLineFromFile(inputFile,strStream);
                strStream >> m_controlParams.maxIt;
                strStream >> m_controlParams.relax1;
                strStream >> m_controlParams.relax1_t;
            } else if (line.compare(m_advancedHeader)==0)
            {
                m_getLineFromFile(inputFile,strStream);
                strStream >> m_controlParams.mapTol;
                strStream >> m_controlParams.venkatK;
                strStream >> m_controlParams.mdotRelax;
                m_getLineFromFile(inputFile,strStream);
                strStream >> m_controlParams.linSolMaxIt;
                strStream >> m_controlParams.linSolTol;
                strStream >> m_controlParams.bigNumber;
            }
        }
    } else
    {
        return 1;
    }
    inputFile.close();
    m_filePath = filePath;
    m_dataIsReady = true;
    return 0;
}

bool FlowParamManager::dataIsReady() const
{
    return m_dataIsReady;
}

inline void FlowParamManager::m_getLineFromFile(ifstream& file, stringstream& stream) const
{
    string line;
    getline(file,line);
    stream.clear();
    stream.str(line);
}

}
