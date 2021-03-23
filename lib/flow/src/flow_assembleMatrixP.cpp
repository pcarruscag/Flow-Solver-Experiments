//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
int PressureBasedCoupledSolver::m_assembleMatrixP(const int prt,
                                                  const int Nf,
                                                  const int Nb)
{
    auto& faces = (*m_part)[prt].faces;
    auto& cells = (*m_part)[prt].cells;

    int C0, C1, bndType;
    float anb, JsfP, Vf, af;
    Vector3f nabla_pf, r1_prime;

    // Internal faces
    for(int i=Nb; i<Nf; i++) m_pressureCoefficients(prt,i);

    // Boundary faces
    for(int i=0; i<Nb; i++)
    {
        C0 = faces.connectivity[i].first;

        auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));

        bndType = m_boundaries[prt].type(i);
        if(bndType == BoundaryType::VELOCITY || bndType == BoundaryType::WALL)
        {   // volume flow without grid flux, conservation of V_inertial
            m_source[prt](4*C0+3) -= m_mdot[prt](i)/m_rho;
        }
        else if(bndType == BoundaryType::TOTALPRESSURE)
        {
            // NOT IMPLEMENTED YET
        }
        else if(bndType == BoundaryType::PRESSURE || bndType == BoundaryType::MASSFLOW)
        {
            Vf = cells.volume(C0);
            af = B00.diag(2);
            anb = Vf/(af*faces.alfa0(i));
            JsfP = anb*(m_nabla[prt].p.row(C0).dot(faces.r0.row(i))-m_flwFld_F[prt].p(i));

            B00.diag(3) += anb;

            for(int j=0; j<3; j++)
              B00.bRow(j) += faces.area(i,j);

            m_source[prt](4*C0+3) -= JsfP;
        }
        else if(bndType == BoundaryType::PERIODIC && m_boundaries[prt].conditions(i,1) == 1)
        {
            C1 = m_boundaries[prt].conditions(i,0);

            auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));
            auto B01 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,0));
            auto B10 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,1));

            Vf = cells.volume(C0)+cells.volume(C1);
            af = B00.diag(2)+B11.diag(2);
            anb = Vf/(af*(faces.alfa0(i)-faces.alfa1(i)));

            // PP coefficients
            B00.diag(3) += anb;  B01.diag(3) = -anb;
            B10.diag(3) = -anb;  B11.diag(3) += anb;

            // PU PV PW coefficients
            B01.bRow(0) = (1.0f-faces.wf(i))*(
                faces.area(i,0)*m_boundaries[prt].conditions(i,2)+
                faces.area(i,1)*m_boundaries[prt].conditions(i,3));
            B01.bRow(1) = (1.0f-faces.wf(i))*(
                faces.area(i,1)*m_boundaries[prt].conditions(i,2)-
                faces.area(i,0)*m_boundaries[prt].conditions(i,3));
            B01.bRow(2) = faces.area(i,2)*(1.0f-faces.wf(i));

            for(int j=0; j<3; j++)
            {
                B10.bRow(j) = -faces.area(i,j)*faces.wf(i);
                B00.bRow(j) -= B10.bRow(j);
                B11.bRow(j) -= B01.bRow(j);
            }

            // Source term
            // rotated C1 pressure gradient
            nabla_pf << m_nabla[prt].p(C1,0)*m_boundaries[prt].conditions(i,2)-
                        m_nabla[prt].p(C1,1)*m_boundaries[prt].conditions(i,3),
                        m_nabla[prt].p(C1,0)*m_boundaries[prt].conditions(i,3)+
                        m_nabla[prt].p(C1,1)*m_boundaries[prt].conditions(i,2),
                        m_nabla[prt].p(C1,2);
            nabla_pf *= (1.0f-faces.wf(i));
            nabla_pf += faces.wf(i)*m_nabla[prt].p.row(C0);

            // rotated r1 vector
            r1_prime << faces.r1(i,0)*m_boundaries[prt].conditions(i,2)-
                        faces.r1(i,1)*m_boundaries[prt].conditions(i,3),
                        faces.r1(i,0)*m_boundaries[prt].conditions(i,3)+
                        faces.r1(i,1)*m_boundaries[prt].conditions(i,2),
                        faces.r1(i,2);

            JsfP = anb*(nabla_pf.dot(faces.r0.row(i))-nabla_pf.dot(r1_prime));

            m_source[prt](4*C0+3) -= JsfP;
            m_source[prt](4*C1+3) += JsfP;
        }
        else if(bndType == BoundaryType::GHOST)
        {
            C1 = m_boundaries[prt].conditions(i,0); // Index of ghost cell

            nabla_pf = faces.wf(i) *m_nabla[prt].p.row(C0)+
                 (1.0f-faces.wf(i))*m_ghostCells[prt].nabla.p.row(C1);

            Vf = cells.volume(C0)+m_ghostCells[prt].vol_nb(C1);
            af = B00.diag(2)+m_ghostCells[prt].diag_nb(C1);
            anb = Vf/(af*(faces.alfa0(i)-faces.alfa1(i)));
            JsfP = anb*nabla_pf.dot(faces.r0.row(i)-faces.r1.row(i));

            B00.diag(3) += anb;
            m_ghostCells[prt].coefMat[C1](3,3) = -anb;
            m_source[prt](4*C0+3) -= JsfP;

            for(int j=0; j<3; j++)
            {
                B00.bRow(j) += faces.area(i,j)*faces.wf(i);
                m_ghostCells[prt].coefMat[C1](3,j) = faces.area(i,j)*(1.0f-faces.wf(i));
            }
        }
    }
    return 0;
}

void PressureBasedCoupledSolver::m_pressureCoefficients(const int prt, const int i)
{
    auto& faces = (*m_part)[prt].faces;
    auto& cells = (*m_part)[prt].cells;

    int C0 = faces.connectivity[i].first;
    int C1 = faces.connectivity[i].second;

    // Fetch matrix blocks
    auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));
    auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));
    auto B01 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,0));
    auto B10 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,1));

    Vector3f nabla_pf = faces.wf(i) *m_nabla[prt].p.row(C0)+
                  (1.0f-faces.wf(i))*m_nabla[prt].p.row(C1);

    float Vf = cells.volume(C0)+cells.volume(C1),
          af = B00.diag(2)+B11.diag(2),
          anb = Vf/(af*(faces.alfa0(i)-faces.alfa1(i))),
          JsfP = anb*nabla_pf.dot(faces.r0.row(i)-faces.r1.row(i));

    // PP coefficients
    B00.diag(3) += anb;  B01.diag(3) = -anb;
    B10.diag(3) = -anb;  B11.diag(3) += anb;

    // PU PV PW coefficients
    for(int j=0; j<3; j++)
    {
        B01.bRow(j) =  faces.area(i,j)*(1.0f-faces.wf(i));
        B10.bRow(j) = -faces.area(i,j)*faces.wf(i);
        B00.bRow(j) -= B10.bRow(j);
        B11.bRow(j) -= B01.bRow(j);
    }

    // Pressure source term
    m_source[prt](4*C0+3) -= JsfP;
    m_source[prt](4*C1+3) += JsfP;
}
}
