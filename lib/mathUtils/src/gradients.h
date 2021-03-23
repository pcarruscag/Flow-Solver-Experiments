//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace mathUtils
{
  namespace gradients
  {
    using namespace Eigen;

    // the Green-Gauss method needs to know whether a face is internal, therefore
    // affecting 2 cells, or is a boundary, affecting only 1 cell, these are helper
    // classes whose method (faceIdx) must return TRUE for boundary faces and v.v.
    // the way to determine this is different depending on data structure organization
    class NumBasedBoundaryIndicator
    {
      // assume the first "bndNum" faces are boundaries
      private:
        NumBasedBoundaryIndicator();
        int m_bndNum;
      public:
        NumBasedBoundaryIndicator(const int bndNum) : m_bndNum(bndNum) {}
        inline bool operator()(const int idx) const {return idx<m_bndNum;}
    };
    class TypeBasedBoundaryIndicator
    {
      // rely on faces to be labeled
      private:
        TypeBasedBoundaryIndicator();
        const ArrayXi* m_faceType;
        int m_code;
      public:
        TypeBasedBoundaryIndicator(const ArrayXi* faceType, const int internalCode)
          : m_faceType(faceType), m_code(internalCode) {}
        inline bool operator()(const int idx) const {return (*m_faceType)(idx)!=m_code;}
    };

    // computation of cell gradients via the divergence theorem
    // all operations done index by index to ensure correct automatic differentiation
    template<typename GeoMat_t, typename GeoVec_t, typename FlwVec_t,
             typename OutVec_t, typename OutMat_t, typename BndInd_t>
    inline void greenGauss(const int Nc, const int Nf,
                           const std::vector<std::pair<int,int> >& connectivity,
                           const BndInd_t& boundaryIndicator,
                           const Ref<const GeoMat_t> area,
                           const Ref<const GeoVec_t> wf,
                           const Ref<const GeoVec_t> volume,
                           const Ref<const FlwVec_t> phiC,
                           Ref<OutVec_t> phiF,
                           Ref<OutMat_t> nabla_phi)
    {

        for(int i=0; i<Nc; ++i) for(int j=0; j<3; ++j) nabla_phi(i,j)=0.0;

        typename OutVec_t::Scalar flux[3];

        for(int i=0; i<Nf; ++i)
        {
            int C0 = connectivity[i].first,
                C1 = connectivity[i].second;
            bool internal = !boundaryIndicator(i);

            if(internal) phiF(i) = wf(i)*phiC(C0)+(1.0f-wf(i))*phiC(C1);

            for(int j=0; j<3; ++j) flux[j] = phiF(i)*area(i,j);

            for(int j=0; j<3; ++j) nabla_phi(C0,j) += flux[j];

            if(internal) for(int j=0; j<3; ++j) nabla_phi(C1,j) -= flux[j];
        }
        for(int i=0; i<Nc; ++i) nabla_phi.row(i) /= volume(i);
    }
  }
}

#endif // GRADIENTS_H
