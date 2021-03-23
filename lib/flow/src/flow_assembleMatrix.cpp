//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
int PressureBasedCoupledSolver::m_assembleMatrix(const int prt,
                                                 const int Nc,
                                                 const int Nf,
                                                 const int Nb,
                                                 const float cfl)
{
    using std::sqrt; using std::cbrt; using std::pow; using std::max;

    auto& faces = (*m_part)[prt].faces;
    auto& cells = (*m_part)[prt].cells;

    int C0, C1, bndType;
    float anb, Jsf[3], JsfP, mdot, mut;

    // Reset matrix, diagonals are build incrementally
    m_coefMat[prt].setZero();

    // Internal faces, this loop cannot be parallel
    for(int i=Nb; i<Nf; i++) m_momentumCoefficients(prt,i);

    // Boundary faces
    for(int i=0; i<Nb; i++)
    {
        C0 = faces.connectivity[i].first;

        auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));

        mdot = m_mdot[prt](i);
        anb = (m_mu+m_mut[prt](C0))/faces.alfa0(i);

        Jsf[0] = -anb*m_nabla[prt].u.row(C0).dot(faces.d0.row(i));
        Jsf[1] = -anb*m_nabla[prt].v.row(C0).dot(faces.d0.row(i));
        Jsf[2] = -anb*m_nabla[prt].w.row(C0).dot(faces.d0.row(i));
        JsfP   =      m_nabla[prt].p.row(C0).dot(faces.d0.row(i));

        bndType = m_boundaries[prt].type(i);
        if(bndType == BoundaryType::VELOCITY || bndType == BoundaryType::WALL)
        {
            for(int k=0; k<3; ++k)
            {
                B00.diag(k) += anb-mdot;
                B00.rCol(k) += faces.area(i,k);
            }
            m_source[prt](4*C0  ) += Jsf[0]-JsfP*faces.area(i,0)+(anb-mdot)*m_flwFld_F[prt].u(i);
            m_source[prt](4*C0+1) += Jsf[1]-JsfP*faces.area(i,1)+(anb-mdot)*m_flwFld_F[prt].v(i);
            m_source[prt](4*C0+2) += Jsf[2]-JsfP*faces.area(i,2)+(anb-mdot)*m_flwFld_F[prt].w(i);
        }
        else if(bndType == BoundaryType::TOTALPRESSURE)
        {
            // NOT IMPLEMENTED YET
        }
        else if(bndType == BoundaryType::PRESSURE || bndType == BoundaryType::MASSFLOW)
        {
            for(int k=0; k<3; ++k)
              m_source[prt](4*C0+k) -= m_flwFld_F[prt].p(i)*faces.area(i,k);
        }
        else if(bndType == BoundaryType::SYMMETRY)
        {
            for(int k=0; k<3; ++k)
            {
                B00.rCol(k) += faces.area(i,k);
                m_source[prt](4*C0+k) += Jsf[k]-JsfP*faces.area(i,k);
            }
        }
        else if(bndType == BoundaryType::PERIODIC && m_boundaries[prt].conditions(i,1) == 1.0f)
        {
            C1 = m_boundaries[prt].conditions(i,0);

            auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));
            auto B01 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,0));
            auto B10 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,1));

            // Diffusion coefficient
            mut = m_mut[prt](C0)*faces.wf(i)+m_mut[prt](C1)*(1.0f-faces.wf(i));
            anb = (m_mu+mut)/(faces.alfa0(i)-faces.alfa1(i));

            // DISCRETIZATION OF FLUXES FOR CELL C0

            // ### DIFFUSION - central differences ###
            // Fluxes for gradient correction
            Jsf[0] = anb*(m_nabla[prt].u.row(C1).dot(faces.d1.row(i))*m_boundaries[prt].conditions(i,2)-
                          m_nabla[prt].v.row(C1).dot(faces.d1.row(i))*m_boundaries[prt].conditions(i,3)-
                          m_flwFld_C[prt].v(C1)*m_boundaries[prt].conditions(i,3)-
                          m_nabla[prt].u.row(C0).dot(faces.d0.row(i)));
            Jsf[1] = anb*(m_nabla[prt].u.row(C1).dot(faces.d1.row(i))*m_boundaries[prt].conditions(i,3)+
                          m_flwFld_C[prt].u(C1)*m_boundaries[prt].conditions(i,3)+
                          m_nabla[prt].v.row(C1).dot(faces.d1.row(i))*m_boundaries[prt].conditions(i,2)-
                          m_nabla[prt].v.row(C0).dot(faces.d0.row(i)));
            Jsf[3] = anb*(m_nabla[prt].w.row(C1).dot(faces.d1.row(i))-
                          m_nabla[prt].w.row(C0).dot(faces.d0.row(i)));

            // Main diagonals of UU VV WW for cell C0
            for(int k=0; k<3; ++k) B00.diag(k) += anb;

            // Neighbours, partial decoupling of u,v due to rotation, acounted in Jsf
            B01.diag(0) = -anb*m_boundaries[prt].conditions(i,2);
            B01.diag(1) = -anb*m_boundaries[prt].conditions(i,2);
            B01.diag(2) = -anb;

            // ### ADVECTION - second order upwind ###
            if(mdot > 0.0f) // Upwind
            {
                Jsf[0] -= mdot*m_limit[prt].u(C0)*m_nabla[prt].u.row(C0).dot(faces.r0.row(i));
                Jsf[1] -= mdot*m_limit[prt].v(C0)*m_nabla[prt].v.row(C0).dot(faces.r0.row(i));
                Jsf[2] -= mdot*m_limit[prt].w(C0)*m_nabla[prt].w.row(C0).dot(faces.r0.row(i));

            } else
            {
                // Main diagonal
                for(int k=0; k<3; ++k) B00.diag(k) -= mdot;

                // Neighbours, partial decoupling of u,v due to rotation, acounted in Jsf
                B01.diag(0) += mdot*m_boundaries[prt].conditions(i,2);
                B01.diag(1) += mdot*m_boundaries[prt].conditions(i,2);
                B01.diag(2) += mdot;

                Jsf[0] -= mdot*(m_limit[prt].u(C1)*m_nabla[prt].u.row(C1).dot(faces.r1.row(i))*
                          m_boundaries[prt].conditions(i,2)-m_boundaries[prt].conditions(i,3)*
                         (m_flwFld_C[prt].v(C1)+m_limit[prt].v(C1)*m_nabla[prt].v.row(C1).dot(faces.r1.row(i))));
                Jsf[1] -= mdot*(m_limit[prt].v(C1)*m_nabla[prt].v.row(C1).dot(faces.r1.row(i))*
                          m_boundaries[prt].conditions(i,2)+m_boundaries[prt].conditions(i,3)*
                         (m_flwFld_C[prt].u(C1)+m_limit[prt].u(C1)*m_nabla[prt].u.row(C1).dot(faces.r1.row(i))));
                Jsf[2] -= mdot*m_limit[prt].w(C1)*m_nabla[prt].w.row(C1).dot(faces.r1.row(i));
            }

            // Source terms (advection and diffusion)
            for(int k=0; k<3; ++k) m_source[prt](4*C0+k) += Jsf[k];

            // DISCRETIZATION OF FLUXES FOR CELL C1

            // ### DIFFUSION - central differences ###
            // Fluxes for gradient correction
            Jsf[0] = anb*(m_nabla[prt].u.row(C0).dot(faces.d0.row(i))*m_boundaries[prt].conditions(i,2)+
                          m_nabla[prt].v.row(C0).dot(faces.d0.row(i))*m_boundaries[prt].conditions(i,3)+
                          m_flwFld_C[prt].v(C0)*m_boundaries[prt].conditions(i,3)-
                          m_nabla[prt].u.row(C1).dot(faces.d1.row(i)));
            Jsf[1] = anb*(m_nabla[prt].u.row(C0).dot(faces.d0.row(i))*m_boundaries[prt].conditions(i,3)*-1.0f-
                          m_flwFld_C[prt].u(C0)*m_boundaries[prt].conditions(i,3)+
                          m_nabla[prt].v.row(C0).dot(faces.d0.row(i))*m_boundaries[prt].conditions(i,2)-
                          m_nabla[prt].v.row(C1).dot(faces.d1.row(i)));
            Jsf[3] = anb*(m_nabla[prt].w.row(C0).dot(faces.d0.row(i))-
                          m_nabla[prt].w.row(C1).dot(faces.d1.row(i)));

            // Main diagonals of UU VV WW for cell C1
            for(int k=0; k<3; ++k) B11.diag(k) += anb;

            // Neighbours, partial decoupling of u,v due to rotation, acounted in Jsf
            B10.diag(0) = -anb*m_boundaries[prt].conditions(i,2);
            B10.diag(1) = -anb*m_boundaries[prt].conditions(i,2);
            B10.diag(2) = -anb;

            // ### ADVECTION - second order upwind ###
            if(mdot > 0.0f) // Upwind
            {
                // Main diagonal
                for(int k=0; k<3; ++k) B11.diag(k) += mdot;

                // Neighbours, partial decoupling of u,v due to rotation, acounted in Jsf
                B10.diag(0) -= mdot*m_boundaries[prt].conditions(i,2);
                B10.diag(1) -= mdot*m_boundaries[prt].conditions(i,2);
                B10.diag(2) -= mdot;

                Jsf[0] += mdot*(m_limit[prt].u(C0)*m_nabla[prt].u.row(C0).dot(faces.r0.row(i))*
                          m_boundaries[prt].conditions(i,2)+m_boundaries[prt].conditions(i,3)*
                         (m_flwFld_C[prt].v(C0)+m_limit[prt].v(C0)*m_nabla[prt].v.row(C0).dot(faces.r0.row(i))));
                Jsf[1] += mdot*(m_limit[prt].v(C0)*m_nabla[prt].v.row(C0).dot(faces.r0.row(i))*
                          m_boundaries[prt].conditions(i,2)-m_boundaries[prt].conditions(i,3)*
                         (m_flwFld_C[prt].u(C0)+m_limit[prt].u(C0)*m_nabla[prt].u.row(C0).dot(faces.r0.row(i))));
                Jsf[2] += mdot*m_limit[prt].w(C0)*m_nabla[prt].w.row(C0).dot(faces.r0.row(i));
            } else
            {
                Jsf[0] += mdot*m_limit[prt].u(C1)*m_nabla[prt].u.row(C1).dot(faces.r1.row(i));
                Jsf[1] += mdot*m_limit[prt].v(C1)*m_nabla[prt].v.row(C1).dot(faces.r1.row(i));
                Jsf[2] += mdot*m_limit[prt].w(C1)*m_nabla[prt].w.row(C1).dot(faces.r1.row(i));
            }

            // Source terms (advection and diffusion)
            for(int k=0; k<3; ++k) m_source[prt](4*C1+k) += Jsf[k];

            // ### PRESSURE ###
            // C0
            for(int k=0; k<3; ++k)
            {
                B00.rCol(k) += faces.area(i,k)*faces.wf(i);
                B01.rCol(k)  = faces.area(i,k)*(1.0f-faces.wf(i));
            }

            // C1
            float Ax = faces.area(i,0)*m_boundaries[prt].conditions(i,2)+
                       faces.area(i,1)*m_boundaries[prt].conditions(i,3),
                  Ay = faces.area(i,1)*m_boundaries[prt].conditions(i,2)-
                       faces.area(i,0)*m_boundaries[prt].conditions(i,3);
            B11.rCol(0) -= Ax*(1.0f-faces.wf(i));
            B11.rCol(1) -= Ay*(1.0f-faces.wf(i));
            B11.rCol(2) -= faces.area(i,2)*(1.0f-faces.wf(i));
            B10.rCol(0)  = -Ax*faces.wf(i);
            B10.rCol(1)  = -Ay*faces.wf(i);
            B10.rCol(2)  = -faces.area(i,2)*faces.wf(i);
        }
        else if(bndType == BoundaryType::GHOST)
        {
            C1 = m_boundaries[prt].conditions(i,0); // Index of ghost cell

            // Diffusion
            mut = m_mut[prt](C0)*faces.wf(i)+m_ghostCells[prt].mut(C1)*(1.0f-faces.wf(i));
            anb = (m_mu+mut)/(faces.alfa0(i)-faces.alfa1(i));

            Jsf[0] = anb*(m_ghostCells[prt].nabla.u.row(C1).dot(faces.d1.row(i))-
                                     m_nabla[prt].u.row(C0).dot(faces.d0.row(i)));
            Jsf[1] = anb*(m_ghostCells[prt].nabla.v.row(C1).dot(faces.d1.row(i))-
                                     m_nabla[prt].v.row(C0).dot(faces.d0.row(i)));
            Jsf[2] = anb*(m_ghostCells[prt].nabla.w.row(C1).dot(faces.d1.row(i))-
                                     m_nabla[prt].w.row(C0).dot(faces.d0.row(i)));

            for(int j=0; j<3; j++)
            {
                B00.diag(j) += anb;
                m_ghostCells[prt].coefMat[C1](j,j) = -anb;
            }

            // Advection
            if(mdot > 0.0f)
            {
                Jsf[0] -= mdot*m_limit[prt].u(C0)*m_nabla[prt].u.row(C0).dot(faces.r0.row(i));
                Jsf[1] -= mdot*m_limit[prt].v(C0)*m_nabla[prt].v.row(C0).dot(faces.r0.row(i));
                Jsf[2] -= mdot*m_limit[prt].w(C0)*m_nabla[prt].w.row(C0).dot(faces.r0.row(i));
            } else
            {
                for(int j=0; j<3; j++)
                {
                    B00.diag(j) -= mdot;
                    m_ghostCells[prt].coefMat[C1](j,j) += mdot;
                }

                Jsf[0] -= mdot*m_ghostCells[prt].limit.u(C1)*
                               m_ghostCells[prt].nabla.u.row(C1).dot(faces.r1.row(i));
                Jsf[1] -= mdot*m_ghostCells[prt].limit.v(C1)*
                               m_ghostCells[prt].nabla.v.row(C1).dot(faces.r1.row(i));
                Jsf[2] -= mdot*m_ghostCells[prt].limit.w(C1)*
                               m_ghostCells[prt].nabla.w.row(C1).dot(faces.r1.row(i));
            }

            // Pressure
            for(int j=0; j<3; j++)
            {
                B00.rCol(j) += faces.area(i,j)*faces.wf(i);
                m_ghostCells[prt].coefMat[C1](j,3) = faces.area(i,j)*(1.0f-faces.wf(i));
            }

            // Source term
            for(int j=0; j<3; j++) m_source[prt](4*C0+j) += Jsf[j];
        }
    }

    // Temporal term
    for(int i=0; i<Nc; ++i)
    {
        float d = cells.volume(i)*m_rho/cfl;

        for(int j=0; j<3; j++)
        {
            auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](i));
            B00.diag(j) += d;
            m_source[prt](4*i+j) += d*m_solution[prt](4*i+j);
        }
    }

    return 0;
}

void PressureBasedCoupledSolver::m_momentumCoefficients(const int prt, const int i)
{
    auto& faces = (*m_part)[prt].faces;

    int C0 = faces.connectivity[i].first;
    int C1 = faces.connectivity[i].second;

    // Fetch matrix blocks
    auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));
    auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));
    auto B01 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,0));
    auto B10 = m_coefMat[prt].getBlock(m_matOffMap[prt](i,1));

    // ### DIFFUSION - central differences ###
    // Coefficient
    float mut = m_mut[prt](C0)*faces.wf(i)+m_mut[prt](C1)*(1.0f-faces.wf(i));
    float anb = (m_mu+mut)/(faces.alfa0(i)-faces.alfa1(i));

    // Fluxes for gradient correction
    float Jsf[3] = {anb*(m_nabla[prt].u.row(C1).dot(faces.d1.row(i))-
                         m_nabla[prt].u.row(C0).dot(faces.d0.row(i))),
                    anb*(m_nabla[prt].v.row(C1).dot(faces.d1.row(i))-
                         m_nabla[prt].v.row(C0).dot(faces.d0.row(i))),
                    anb*(m_nabla[prt].w.row(C1).dot(faces.d1.row(i))-
                         m_nabla[prt].w.row(C0).dot(faces.d0.row(i)))};

    // Neighbour coefficients
    for(int j=0; j<3; ++j)
      B01.diag(j) = B10.diag(j) = -anb;

    // ### ADVECTION - second order upwind ###
    // Coefficient
    float mdot = m_mdot[prt](i);

    if(mdot > 0.0f)
    {
        // convected from C0 to C1
        for(int j=0; j<3; j++) B10.diag(j) -= mdot;

        Jsf[0] -= mdot*m_limit[prt].u(C0)*m_nabla[prt].u.row(C0).dot(faces.r0.row(i));
        Jsf[1] -= mdot*m_limit[prt].v(C0)*m_nabla[prt].v.row(C0).dot(faces.r0.row(i));
        Jsf[2] -= mdot*m_limit[prt].w(C0)*m_nabla[prt].w.row(C0).dot(faces.r0.row(i));
    } else
    {
        // convected from C1 to C0
        for(int j=0; j<3; j++) B01.diag(j) += mdot;

        Jsf[0] -= mdot*m_limit[prt].u(C1)*m_nabla[prt].u.row(C1).dot(faces.r1.row(i));
        Jsf[1] -= mdot*m_limit[prt].v(C1)*m_nabla[prt].v.row(C1).dot(faces.r1.row(i));
        Jsf[2] -= mdot*m_limit[prt].w(C1)*m_nabla[prt].w.row(C1).dot(faces.r1.row(i));
    }

    // Main diagonals of UU VV WW (a0 = sum(anb))
    for(int j=0; j<3; ++j)
    {
        B00.diag(j) -= B01.diag(j);
        B11.diag(j) -= B10.diag(j);
    }

    // ### PRESSURE - Green-Gauss with linear interpolation of face value ###
    for(int j=0; j<3; j++)
    {
        B01.rCol(j) =  faces.area(i,j)*(1.0f-faces.wf(i));
        B10.rCol(j) = -faces.area(i,j)*faces.wf(i);
        B00.rCol(j) -= B10.rCol(j);
        B11.rCol(j) -= B01.rCol(j);
    }

    // Update source terms
    for(int j=0; j<3; j++)
    {
        m_source[prt](4*C0+j) += Jsf[j];
        m_source[prt](4*C1+j) -= Jsf[j];
    }
}
}
