//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
template<class ModelType>
int PressureBasedCoupledSolver::m_assembleMatrixT(const int prt,
                                                  const int Nf,
                                                  const int Nb,
                                                  const ModelType* turbModel,
                                                  const turbulenceModels::ModelBase::ModelVariables turbVar)
{
    auto& faces = (*m_part)[prt].faces;
    auto& cells = (*m_part)[prt].cells;

    int C0, C1, bndType;
    float anb, Jsf, mdot, mut;
    float* matrix = m_coefMat_t[prt].valuePtr();

    VectorXf *p_valueF, *p_limit, *p_limitGC;
    MatrixX3f *p_nabla, *p_nablaGC;

    if(turbVar == turbulenceModels::ModelBase::FIRST)
    {
        p_valueF  = &m_flwFld_F[prt].turb1;
        p_limit   = &m_limit[prt].turb1;
        p_limitGC = &m_ghostCells[prt].limit.turb1;
        p_nabla   = &m_nabla[prt].turb1;
        p_nablaGC = &m_ghostCells[prt].nabla.turb1;
    }
    else // turbulenceModels::ModelBase::SECOND
    {
        p_valueF  = &m_flwFld_F[prt].turb2;
        p_limit   = &m_limit[prt].turb2;
        p_limitGC = &m_ghostCells[prt].limit.turb2;
        p_nabla   = &m_nabla[prt].turb2;
        p_nablaGC = &m_ghostCells[prt].nabla.turb2;
    }

    // Reset matrix diagonal
    turbModel->linearizedSourceTerm(turbVar, m_matDiagMap[prt].data(),
                                    cells.volume.data(), matrix);

    // Internal faces and periodics. This loop cannot be parallel or simd
    for(int i=0; i<Nf; i++)
    {
        C0 = faces.connectivity[i].first;
        C1 = faces.connectivity[i].second;

        if(i<Nb) {
          if(m_boundaries[prt].type(i) != BoundaryType::PERIODIC ||
             m_boundaries[prt].conditions(i,1) != 1) continue;

          C1 = m_boundaries[prt].conditions(i,0);
        }

        auto& B00 = matrix[m_matDiagMap[prt](C0)];
        auto& B11 = matrix[m_matDiagMap[prt](C1)];
        auto& B01 = matrix[m_matOffMap[prt](i,0)];
        auto& B10 = matrix[m_matOffMap[prt](i,1)];

        mdot = m_mdot[prt](i);

        mut = m_mut[prt](C0)*turbModel->viscosityMultiplier(turbVar,C0)*faces.wf(i)+
              m_mut[prt](C1)*turbModel->viscosityMultiplier(turbVar,C1)*(1.0f-faces.wf(i));
        anb = (m_mu+mut)/(faces.alfa0(i)-faces.alfa1(i));
        Jsf = anb*((*p_nabla).row(C1).dot(faces.d1.row(i))-
                   (*p_nabla).row(C0).dot(faces.d0.row(i)));

        B01 = B10 = -anb;

        if(mdot > 0.0f) {
            B10 -= mdot;
            Jsf -= mdot*(*p_limit)(C0)*(*p_nabla).row(C0).dot(faces.r0.row(i));
        } else {
            B01 += mdot;
            Jsf -= mdot*(*p_limit)(C1)*(*p_nabla).row(C1).dot(faces.r1.row(i));
        }
        B00 -= B01;
        B11 -= B10;
        m_source_t[prt](C0) += Jsf;
        m_source_t[prt](C1) -= Jsf;
    }

    // Boundary faces
    for(int i=0; i<Nb; i++)
    {
        C0 = faces.connectivity[i].first;

        auto& B00 = matrix[m_matDiagMap[prt](C0)];

        mdot = m_mdot[prt](i);

        mut = m_mut[prt](C0)*turbModel->viscosityMultiplier(turbVar,C0);
        anb = (m_mu+mut)/faces.alfa0(i);
        Jsf = -anb*(*p_nabla).row(C0).dot(faces.d0.row(i));

        bndType = m_boundaries[prt].type(i);
        if(bndType == BoundaryType::VELOCITY || bndType == BoundaryType::TOTALPRESSURE)
        {
            B00 += anb-mdot;
            m_source_t[prt](C0) += Jsf+(anb-mdot)*(*p_valueF)(i);
        }
        else if(bndType == BoundaryType::PRESSURE || bndType == BoundaryType::MASSFLOW)
        {
//            m_source_t[prt](C0) -= mdot*(*p_limit)(C0)*(*p_nabla).row(C0).dot(faces.d0.row(i));
        }
        else if(bndType == BoundaryType::WALL)
        {
            if(turbVar == turbulenceModels::ModelBase::FIRST) {
                if(turbModel->firstVarWallBC() == turbulenceModels::ModelBase::DIRICHLET) {
                    B00 += anb;
                    m_source_t[prt](C0) += Jsf+anb*(*p_valueF)(i);
                }
            } else {
                B00 += m_control.bigNumber;
                m_source_t[prt](C0) += m_control.bigNumber*(*p_valueF)(i);
            }
        }
        else if(bndType == BoundaryType::SYMMETRY)
        {
            m_source_t[prt](C0) += Jsf;
        }
        else if(bndType == BoundaryType::GHOST)
        {
            C1 = m_boundaries[prt].conditions(i,0); // Index of ghost cell

            mut = (m_mut[prt](C0)*faces.wf(i)+m_ghostCells[prt].mut(C1)*(1.0f-faces.wf(i)))*
                   turbModel->viscosityMultiplier(turbVar,C0);
            anb = (m_mu+mut)/(faces.alfa0(i)-faces.alfa1(i));
            Jsf = anb*((*p_nablaGC).row(C1).dot(faces.d1.row(i))-(*p_nabla).row(C0).dot(faces.d0.row(i)));

            m_ghostCells[prt].coefMat_t[C1](0,0) = -anb;

            if(mdot > 0.0f)
            {
                Jsf -= mdot*(*p_limit)(C0)*(*p_nabla).row(C0).dot(faces.r0.row(i));
            } else
            {
                m_ghostCells[prt].coefMat_t[C1](0,0) += mdot;
                Jsf -= mdot*(*p_limitGC)(C1)*(*p_nablaGC).row(C1).dot(faces.r1.row(i));
            }
            B00 -= m_ghostCells[prt].coefMat_t[C1](0,0);
            m_source_t[prt](C0) += Jsf;
        }
    }
    return 0;
}
template int PressureBasedCoupledSolver::m_assembleMatrixT<turbulenceModels::MenterSST>(const int,
  const int,const int,const turbulenceModels::MenterSST*,const turbulenceModels::ModelBase::ModelVariables);
}
