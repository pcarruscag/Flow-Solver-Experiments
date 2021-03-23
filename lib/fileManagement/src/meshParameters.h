//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef MESHPARAMETERSMANAGER_H
#define MESHPARAMETERSMANAGER_H

#include <string>
#include <vector>

// Forward declaration of friend classes
namespace geometryGeneration
{
    class BladeGeometry;
}
namespace mesh
{
    class PassageMesh;
}

namespace fileManagement
{
    class MeshParamManager
    {
        friend class geometryGeneration::BladeGeometry;
        friend class mesh::PassageMesh;
      public:
        struct m_orthogCtrl
        {
            std::vector<double> CP;
            double angle;
            double weight;
        };
        struct edgeParams
        {
            int nCellBld;
            double alphaBld;
            double alphaLeTe;
        };
        struct layerParams
        {
            int nCellPitch;
            double nearWallSize;
            double nearInOutSizeRatio;
            std::vector<m_orthogCtrl> orthogCtrl;
        };
        struct volumeParams
        {
            int nCellSpan;
            double nearWallSize;
        };
        struct advancedParams
        {
            double decayFactor;
            double disableOrthogAlfa;
            double disableOrthogFraction;
            int nMultiGrid;
            int iterMax;
            double tol;
            double relax0;
            double relax1;
            int iter1;
            int linSolItMax;
            double linSolTol;
        };
        MeshParamManager();
        int readFile(const std::string &filePath);
        bool dataIsReady() const;
        bool getEdgeParams(edgeParams &outData) const;
        bool getLayerParams(layerParams &outData) const;
        bool getVolParams(volumeParams &outData) const;
        bool getAdvParams(advancedParams &outData) const;
      private:
        bool m_dataIsReady;
        std::string m_filePath;
        edgeParams m_edgePar;
        layerParams m_layerPar;
        volumeParams m_volumePar;
        advancedParams m_advPar;

        std::string m_edgeHeader = "### Edge Mesh Parameters ###";
        std::string m_layerHeader = "### Layer Mesh Parameters ###";
        std::string m_volumeHeader = "### Volume Mesh Parameters ###";
        std::string m_advancedHeader = "### Advanced Parameters ###";
    };
}


#endif // MESHPARAMETERSMANAGER_H
