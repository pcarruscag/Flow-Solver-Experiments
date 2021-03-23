//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef GEOMETRYPARAMETERSMANAGER_H
#define GEOMETRYPARAMETERSMANAGER_H

#include <vector>
#include <string>

// Forward declaration of friend classes
namespace geometryGeneration
{
    class BladeGeometry;
}
namespace mesh
{
    class PassageMeshInspector;
}

namespace fileManagement
{
    class GeometryParamManager
    {
        friend class geometryGeneration::BladeGeometry;
        friend class mesh::PassageMeshInspector;
      public:
        struct advancedParams
        {
            int nPtMeriIntegral;
            int downsampleFactor;
            int nPtBldSurfInt;
        };
        struct geometryInfo
        {
            int nCPhub, nCPshroud, nLayers, nCPtheta;
            geometryInfo(int nh, int ns, int nl, int nt) :
                nCPhub(nh),nCPshroud(ns),nLayers(nl),nCPtheta(nt){}
        };
        GeometryParamManager();
        int readFile(const std::string &filePath);
        bool dataIsReady() const;
        bool getNblades(int &n) const;
        bool getHub(std::vector<double> &x,
                    std::vector<double> &y) const;
        bool getShroud(std::vector<double> &x,
                       std::vector<double> &y) const;
        bool getLe(std::vector<double> &y) const;
        bool getTe(std::vector<double> &y) const;
        bool getTheta(std::vector<std::vector<double> > &y) const;
        bool getThick(std::vector<std::vector<double> > &x,
                      std::vector<std::vector<double> > &y) const;
        bool getAdvanced(advancedParams &outData) const;
        geometryInfo getGeometryInfo() const {
            return geometryInfo(m_nPtHub,m_nPtShr,m_nSpanSec,m_nPtTheta);
        }
      private:
        bool m_dataIsReady;
        std::string m_filePath;
        int m_nBlades;
        int m_nPtHub, m_nPtShr;
        std::vector<double> m_xHubPt, m_yHubPt, m_xShrPt, m_yShrPt;
        int m_nSpanSec;
        std::vector<double> m_yLePt, m_yTePt;
        int m_nPtTheta;
        std::vector< std::vector<double> > m_yThetaPt;
        int m_nPtThick;
        std::vector< std::vector<double> > m_xThickPt, m_yThickPt;
        advancedParams m_advPar;

        std::string m_nBldHeader = "### Number of Blades ###";
        std::string m_hubPtHeader = "### Hub Control Points ###";
        std::string m_shrPtHeader = "### Shroud Control Points ###";
        std::string m_leTePtHeader = "### Leading and Trailing Edge Control Points ###";
        std::string m_thetaPtHeader = "### Wrap Angle Control Points ###";
        std::string m_thickPtHeader = "### Thickness Control Points ###";
        std::string m_advancedHeader = "### Advanced Parameters ###";
    };

}

#endif // GEOMETRYPARAMETERSMANAGER_H
