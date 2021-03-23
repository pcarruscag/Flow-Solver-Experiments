//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "geometryParameters.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace fileManagement{

GeometryParamManager::GeometryParamManager()
{
    m_dataIsReady=false;
}

int GeometryParamManager::readFile(const std::string &filePath)
{
    std::ifstream inputFile;
    std::string line;
    std::stringstream strStream;

    inputFile.open(filePath.c_str());
    if (inputFile.is_open())
    {
        getline(inputFile,line);
        if (line.compare(m_nBldHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nBlades;
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_hubPtHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nPtHub;
            m_xHubPt.resize(m_nPtHub);
            m_yHubPt.resize(m_nPtHub);
            for (int i=0; i<m_nPtHub; i++)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_xHubPt[i] >> m_yHubPt[i];
            }
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_shrPtHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nPtShr;
            m_xShrPt.resize(m_nPtShr);
            m_yShrPt.resize(m_nPtShr);
            for (int i=0; i<m_nPtShr; i++)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_xShrPt[i] >> m_yShrPt[i];
            }
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_leTePtHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nSpanSec;
            m_yLePt.resize(m_nSpanSec);
            m_yTePt.resize(m_nSpanSec);
            for (int i=0; i<m_nSpanSec; i++)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                strStream >> m_yLePt[i] >> m_yTePt[i];
            }
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_thetaPtHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nPtTheta;
            m_yThetaPt.resize(m_nSpanSec);
            for (int i=0; i<m_nSpanSec; i++)
                m_yThetaPt[i].resize(m_nPtTheta);
            for (int j=0; j<m_nPtTheta; j++)
            {
                getline(inputFile,line);
                strStream.clear();
                strStream.str(line);
                for (int i=0; i<m_nSpanSec; i++)
                    strStream >> m_yThetaPt[i][j];
            }
        } else
        {
            inputFile.close();
            return 2;
        }
        getline(inputFile,line);
        if (line.compare(m_thickPtHeader)==0)
        {
            getline(inputFile,line);
            strStream.clear();
            strStream.str(line);
            strStream >> m_nPtThick;
            m_xThickPt.resize(m_nSpanSec);
            m_yThickPt.resize(m_nSpanSec);
            for (int i=0; i<m_nSpanSec; i++)
            {
                m_xThickPt[i].resize(m_nPtThick);
                m_yThickPt[i].resize(m_nPtThick);
                for (int j=0; j<m_nPtThick; j++)
                {
                    getline(inputFile,line);
                    strStream.clear();
                    strStream.str(line);
                    strStream >> m_xThickPt[i][j] >> m_yThickPt[i][j];
                }
            }
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
            strStream >> m_advPar.nPtMeriIntegral;
            strStream >> m_advPar.downsampleFactor;
            strStream >> m_advPar.nPtBldSurfInt;
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

bool GeometryParamManager::dataIsReady() const
{
    return m_dataIsReady;
}

bool GeometryParamManager::getNblades(int &n) const
{
    if (m_dataIsReady)
        n = m_nBlades;
    return m_dataIsReady;
}

bool GeometryParamManager::getHub(std::vector<double> &x,std::vector<double> &y) const
{
    if (m_dataIsReady)
    {
        x = m_xHubPt;
        y = m_yHubPt;
    }
    return m_dataIsReady;
}

bool GeometryParamManager::getShroud(std::vector<double> &x,std::vector<double> &y) const
{
    if (m_dataIsReady)
    {
        x = m_xShrPt;
        y = m_yShrPt;
    }
    return m_dataIsReady;
}

bool GeometryParamManager::getLe(std::vector<double> &y) const
{
    if (m_dataIsReady)
        y = m_yLePt;
    return m_dataIsReady;
}

bool GeometryParamManager::getTe(std::vector<double> &y) const
{
    if (m_dataIsReady)
        y = m_yTePt;
    return m_dataIsReady;
}

bool GeometryParamManager::getTheta(std::vector<std::vector<double> > &y) const
{
    if (m_dataIsReady)
        y = m_yThetaPt;
    return m_dataIsReady;
}

bool GeometryParamManager::getThick(std::vector<std::vector<double> > &x,
                                    std::vector<std::vector<double> > &y) const
{
    if (m_dataIsReady)
    {
        x = m_xThickPt;
        y = m_yThickPt;
    }
    return m_dataIsReady;
}

bool GeometryParamManager::getAdvanced(advancedParams &outData) const
{
    if (m_dataIsReady)
    {
        outData = m_advPar;
    }
    return m_dataIsReady;
}

} // namespace fileManagement


