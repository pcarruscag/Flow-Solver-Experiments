//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"
#include "../../mathUtils/src/matrix.h"

#include <Eigen/Sparse>

namespace flow
{
int PressureBasedCoupledSolver::m_matrixIndexes()
{
    using Triplet = mathUtils::matrix::Triplet<float>;

    m_matDiagMap.resize(m_partNum);
    m_matOffMap.resize(m_partNum);

    m_coefMat.resize(m_partNum);
    m_coefMat_t.resize(m_partNum);

    #pragma omp parallel num_threads(m_partNum)
    {
        int prt = omp_get_thread_num();
        int Nc = (*m_part)[prt].cells.number,
            Nf = (*m_part)[prt].faces.number,
            Nb = m_boundaries[prt].number,
            Npb, Nnz;
        {
            vector<int> periodicBoundaries;
            m_bndrsOfType(prt,7,periodicBoundaries);
            Npb = periodicBoundaries.size();
            Nnz = Nc+2*(Nf-Nb+Npb/2);

            vector<Triplet> coefficients;
            coefficients.reserve(Nnz);

            // Main diagonal
            for(int i=0; i<Nc; i++)
              coefficients.emplace_back(Triplet(i,i,0.0f));

            // Off-diagonal coefficients introduced by periodic boundaries
            for(int i=0; i<Nb; i++)
            {
              if(m_boundaries[prt].type(i)==BoundaryType::PERIODIC &&
                 m_boundaries[prt].conditions(i,1)==1)
              {
                int C0 = (*m_part)[prt].faces.connectivity[i].first;
                int C1 = m_boundaries[prt].conditions(i,0);
                coefficients.emplace_back(Triplet(C0, C1, 0.0f));
                coefficients.emplace_back(Triplet(C1, C0, 0.0f));
              }
            }

            // Off-diagonal coefficients
            for(int i=Nb; i<Nf; i++)
            {
                int C0 = (*m_part)[prt].faces.connectivity[i].first;
                int C1 = (*m_part)[prt].faces.connectivity[i].second;
                coefficients.emplace_back(Triplet(C0, C1, 0.0f));
                coefficients.emplace_back(Triplet(C1, C0, 0.0f));
            }

            // Build turbulence matrix
            m_coefMat_t[prt].resize(Nc,Nc);
            m_coefMat_t[prt].setFromTriplets(coefficients.begin(),coefficients.end());

            assert(Nnz == m_coefMat_t[prt].nonZeros() && "Duplicated matrix entry.");
        } // out of scope triplets before allocating larger matrix

        m_coefMat[prt].setStructure(Nc, Nnz,
                                    m_coefMat_t[prt].outerIndexPtr(),
                                    m_coefMat_t[prt].innerIndexPtr());
        m_coefMat[prt].allocate();

        // Use momentum and pressure matrix to map coefficients
        // Maps are valid for both matrices due to block format
        m_matDiagMap[prt].resize(Nc);
        m_matOffMap[prt].resize(Nf,2);

        for(int i=0; i<Nc; i++)
          m_matDiagMap[prt](i) = m_coefMat[prt].getBlockIndex(i,i);

        for(int i=0; i<Nb; i++)
        {
          if(m_boundaries[prt].type(i)==BoundaryType::PERIODIC &&
             m_boundaries[prt].conditions(i,1)==1)
          {
            int C0 = (*m_part)[prt].faces.connectivity[i].first;
            int C1 = m_boundaries[prt].conditions(i,0);

            m_matOffMap[prt](i,0) = m_coefMat[prt].getBlockIndex(C0,C1);
            m_matOffMap[prt](i,1) = m_coefMat[prt].getBlockIndex(C1,C0);
          }
          else {
            m_matOffMap[prt](i,0) = -1;
            m_matOffMap[prt](i,1) = -1;
          }
        }

        for(int i=Nb; i<Nf; i++)
        {
            int C0 = (*m_part)[prt].faces.connectivity[i].first;
            int C1 = (*m_part)[prt].faces.connectivity[i].second;

            m_matOffMap[prt](i,0) = m_coefMat[prt].getBlockIndex(C0,C1);
            m_matOffMap[prt](i,1) = m_coefMat[prt].getBlockIndex(C1,C0);
        }
    }
    return 0;
}
}
