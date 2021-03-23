//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef PASSAGEMESH_H
#define PASSAGEMESH_H

#include <vector>
#include <iostream>
#include <omp.h>
#include <math.h>

#include "../../fileManagement/src/geometryParameters.h"
#include "../../fileManagement/src/meshParameters.h"
#include "../../geometry/src/bladeGeometry.h"
#include "../../mathUtils/src/matrix.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define TripletD mathUtils::matrix::Triplet<double>

using namespace Eigen;

namespace mesh
{
    class PassageMeshInspector;
    class UnstructuredMesh;

    class PassageMesh
    {
        friend class PassageMeshInspector;
      public:
        // always use sequential numbering starting at 0,
        // used for indexing in convertToUnstructured.
        enum Boundaries {HUB     = 0,
                         SHROUD  = 1,
                         INFLOW  = 2,
                         OUTFLOW = 3,
                         PER_1   = 4,
                         PER_2   = 5,
                         BLADE   = 6};

        PassageMesh(const bool minimumMemory=false);
        int mesh (const geometryGeneration::BladeGeometry &bladeGeo,
                  const fileManagement::MeshParamManager  &meshParams);
        int morph(const geometryGeneration::BladeGeometry &bladeGeo);
        // scale must be called every time after mesh/morph
        int scale(const float factor, const int nMeriVars);
        int convertToUnstructured(UnstructuredMesh& target) const;
        int saveSU2file(const std::string &filePath) const;
        int getGrid(const MatrixXf* &grid_p) const
        {
            if(!m_isMeshed) return 1;
            grid_p = &m_vertexCoords;
            return 0;
        }
        int getJacobians(const MatrixXf* &xCoordJacobian_p,
                         const MatrixXf* &yCoordJacobian_p,
                         const MatrixXf* &zCoordJacobian_p)
        {
            if(!m_isMeshed) return 1;
            if(m_minMemory) {
              // refresh volume mesh and keep the jacobians this time
              m_spanwiseDriver(true);
              if(m_isScaled) scale(m_scale_factor,m_scale_nMeriVars);
            }
            xCoordJacobian_p = &m_xCoordJacobian;
            yCoordJacobian_p = &m_yCoordJacobian;
            zCoordJacobian_p = &m_zCoordJacobian;
            return 0;
        }
        struct meshInfo
        {
            int nMeri, nPitch, nSpan, jLE, jTE, nLayers, nVertex, nVars;
            meshInfo(int nM,int nP,int nS,int jL,int jT,int nL,int nV,int nVar) :
                nMeri(nM),nPitch(nP),nSpan(nS),jLE(jL),jTE(jT),nLayers(nL),nVertex(nV),nVars(nVar){}
        };
        meshInfo getMeshInfo() const
        {
            return meshInfo(m_layerMeri[0].cols(),m_layerMeri[0].rows(),m_nSpanGrdPts,m_jLE,m_jTE,
                            m_layerMeri.size(),m_vertexCoords.rows(),m_xCoordJacobian.cols());
        }
      private:
        // Methods to generate initial H-Mesh of 2D layers
        // Pointers needed in m_restrict because Xc and Yc are resized
        void m_restrict(MatrixXd* Xc, MatrixXd* Yc,
                        const Ref<const MatrixXd> Xf, const Ref<const MatrixXd> Yf, int mgLvl) const;
        int m_meshLayer(Ref<MatrixXd> X, Ref<MatrixXd> Y, const int jLE, const int jTE,
            const int layer, const fileManagement::MeshParamManager &meshParams,
            const bool verbose) const;
        void m_meshLayer_fillStencil(const Ref<const MatrixXd> X,    const Ref<const MatrixXd> Y,
                                     const Ref<const MatrixXd> Xqsi, const Ref<const MatrixXd> Xeta,
                                     const Ref<const MatrixXd> Yqsi, const Ref<const MatrixXd> Yeta,
                                     const Ref<const MatrixXd> J,    const Ref<const MatrixXd> ALFA,
                                     const Ref<const MatrixXd> BETA, const Ref<const MatrixXd> GAMA,
                                     const Ref<const MatrixXd> P_1,  const Ref<const MatrixXd> P_M,
                                     const Ref<const MatrixXd> Q_1,  const Ref<const MatrixXd> Q_M,
                                     const Ref<const VectorXi> LIN,  const Ref<const VectorXi> COL,
                                     const Ref<const MatrixXi> NUM,  const double decayFactor,
                                     const double disableOrthogAlfa, const double disableOrthogFraction,
                                     const double weight, const int jLE, const int jTE,
                                     std::vector<TripletD> &COEF, Ref<MatrixX2d> src) const;
        void m_interpolate(const Ref<const MatrixXd> Xc, const Ref<const MatrixXd> Yc,
                           Ref<MatrixXd> Xf, Ref<MatrixXd> Yf, const int mgLvl) const;
        void m_expansionLayer(Ref<MatrixXd> X, Ref<MatrixXd> Y, const double ds_bld,
                              const double ds_inOut) const;

        // Shared operation pipeline of "mesh" and "morph"
        int m_volumeMesh(const geometryGeneration::BladeGeometry& bladeGeo);
        int m_spanwiseDriver(const bool computeJacobians);

        // Shared operations
        int m_computeLayerJacobian(const Ref<const MatrixXd> X, const Ref<const MatrixXd> Y, const int layerIndex);
        int m_meshSpanwise(const int row, const int col, const int rows, const int cols,
                           const int nSpan, const bool computeJacobians);

        // Reference to the blade geometry used in m_volumeMesh
        const geometryGeneration::BladeGeometry* m_bladeGeo;

        int m_nSpanGrdPts, m_jLE, m_jTE;
        double m_nearWallSize;
        std::vector<MatrixXd> m_layerMeri, m_layerPitch, m_layerJacobian;
        MatrixXf m_vertexCoords;
        MatrixXf m_xCoordJacobian, m_yCoordJacobian, m_zCoordJacobian;
        bool m_isMeshed, m_minMemory;
        // this information is saved for when the jacobians are generated
        bool m_isScaled;
        int  m_scale_nMeriVars;
        float m_scale_factor;
    };

}

#endif // PASSAGEMESH_H
