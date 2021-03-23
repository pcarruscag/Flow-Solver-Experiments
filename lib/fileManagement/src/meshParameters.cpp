//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "meshParameters.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace fileManagement
{
    MeshParamManager::MeshParamManager()
    {
        m_dataIsReady = false;
    }

    bool MeshParamManager::dataIsReady() const
    {
        return m_dataIsReady;
    }

    int MeshParamManager::readFile(const std::string &filePath)
    {
        std::ifstream inputFile;
        std::string line;
        std::stringstream strStream;
        int nSpan;

        inputFile.open(filePath.c_str());
        if (inputFile.is_open())
        {
            getline(inputFile,line);
            if (line.compare(m_edgeHeader)==0)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_edgePar.nCellBld;
                strStream >> m_edgePar.alphaBld;
                strStream >> m_edgePar.alphaLeTe;
            } else
            {
                inputFile.close();
                return 2;
            }
            getline(inputFile,line);
            if (line.compare(m_layerHeader)==0)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_layerPar.nCellPitch;
                strStream >> m_layerPar.nearWallSize;
                strStream >> m_layerPar.nearInOutSizeRatio;
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> nSpan;
                m_layerPar.orthogCtrl.resize(nSpan);
                for(int i=0; i<nSpan; i++) {
                    getline(inputFile,line);
                    strStream.clear();
                    strStream.str(line);
                    m_layerPar.orthogCtrl[i].CP.resize(4);
                    for(int j=0; j<4; j++)
                        strStream >> m_layerPar.orthogCtrl[i].CP[j];
                    strStream >> m_layerPar.orthogCtrl[i].angle;
                    strStream >> m_layerPar.orthogCtrl[i].weight;
                }
            } else
            {
                inputFile.close();
                return 2;
            }
            getline(inputFile,line);
            if (line.compare(m_volumeHeader)==0)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_volumePar.nCellSpan;
                strStream >> m_volumePar.nearWallSize;
            } else
            {
                inputFile.close();
                return 2;
            }
            getline(inputFile,line);
            if (line.compare(m_advancedHeader)==0)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_advPar.decayFactor;
                strStream >> m_advPar.disableOrthogAlfa;
                strStream >> m_advPar.disableOrthogFraction;
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_advPar.nMultiGrid;
                strStream >> m_advPar.iterMax;
                strStream >> m_advPar.tol;
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_advPar.relax0;
                strStream >> m_advPar.relax1;
                strStream >> m_advPar.iter1;
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_advPar.linSolItMax;
                strStream >> m_advPar.linSolTol;
            } else
            {
                inputFile.close();
                return 2;
            }
        } else
        {
            return 1;
        }
        inputFile.close();
        m_filePath=filePath;
        m_dataIsReady=true;
        return 0;
    }

    bool MeshParamManager::getEdgeParams(edgeParams &outData) const
    {
        if (m_dataIsReady)
            outData = m_edgePar;
        return m_dataIsReady;
    }

    bool MeshParamManager::getLayerParams(layerParams &outData) const
    {
        if (m_dataIsReady)
            outData = m_layerPar;
        return m_dataIsReady;
    }

    bool MeshParamManager::getVolParams(volumeParams &outData) const
    {
        if (m_dataIsReady)
            outData = m_volumePar;
        return m_dataIsReady;
    }

    bool MeshParamManager::getAdvParams(advancedParams &outData) const
    {
        if (m_dataIsReady)
            outData = m_advPar;
        return m_dataIsReady;
    }

}
