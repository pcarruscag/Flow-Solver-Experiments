//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjointParameters.h"

#include <cassert>

namespace fileManagement
{
int AdjointParamManager::readFile(const string& filePath)
{
    ifstream inputFile;
    string line;
    stringstream strStream;

    m_dataIsReady = false;

    inputFile.open(filePath.c_str());
    if(!inputFile.good()) return 1;

    getline(inputFile,line);
    if (line.compare(m_objectivesHeader)==0)
    {
        int numObj;
        m_getLineFromFile(inputFile,strStream);
        strStream >> numObj;
        m_objectives.reserve(numObj);
        for(int i=0; i<numObj; ++i)
        {
            getline(inputFile,line);
            int idxObj = 0;
            for(auto name : m_objectiveNames) {
              if(line.compare(name)==0)
                break;
              ++idxObj;
            }
            assert(idxObj<int(m_objectiveNames.size()) && "unrecognized objective");
            m_objectives.push_back(idxObj);
        }
    } else
    {
        inputFile.close();
        return 2;
    }
    getline(inputFile,line);
    if (line.compare(m_convergenceHeader)==0)
    {
        m_getLineFromFile(inputFile,strStream);
        strStream >> m_maxIt;
        strStream >> m_tolerance;
        m_getLineFromFile(inputFile,strStream);
        strStream >> m_alphaMin;
        strStream >> m_alphaMax;
    } else
    {
        inputFile.close();
        return 2;
    }
    getline(inputFile,line);
    if (line.compare(m_advancedHeader)==0)
    {
        getline(inputFile,m_scratchDir);
        m_getLineFromFile(inputFile,strStream);
        strStream >> m_linSolMaxIt;
        strStream >> m_linSolInitTol;
        strStream >> m_linSolMinTol;
        strStream >> m_linSolRatioTol;
        m_getLineFromFile(inputFile,strStream);
        strStream >> m_firstOrder;
        if (!(strStream >> m_rightPrec)) m_rightPrec = false;
    } else
    {
        inputFile.close();
        return 2;
    }
    inputFile.close();
    m_filePath = filePath;
    m_dataIsReady = true;
    return 0;
}

int AdjointParamManager::getObjectives(vector<int>& objectives) const
{
    if(!m_dataIsReady) return 1;
    objectives = m_objectives;
    return 0;
}

int AdjointParamManager::getFirstOrder(bool& firstOrder) const
{
    if(!m_dataIsReady) return 1;
    firstOrder = m_firstOrder;
    return 0;
}

int AdjointParamManager::getRightPrec(bool& rightPrec) const
{
    if(!m_dataIsReady) return 1;
    rightPrec = m_rightPrec;
    return 0;
}

int AdjointParamManager::getScratchDir(string& scratchDir) const
{
    if(!m_dataIsReady) return 1;
    scratchDir = m_scratchDir;
    return 0;
}

inline void AdjointParamManager::m_getLineFromFile(ifstream& file, stringstream& stream) const
{
    string line;
    getline(file,line);
    stream.clear();
    stream.str(line);
}
}
