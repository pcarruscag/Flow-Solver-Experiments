//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
void PressureBasedCoupledSolver::m_boundaryValues(const int prt, const turbulenceModels::ModelBase* turbModel)
{
    using std::pow; using std::sqrt; using std::log; using std::max;

    auto& faces = (*m_part)[prt].faces;
    auto& nabla = m_nabla[prt];
    auto& limit = m_limit[prt];
    auto& flowC = m_flwFld_C[prt];
    auto& flowF = m_flwFld_F[prt];
    auto& flowGC = m_ghostCells[prt].flwFld;
    auto& boundary = m_boundaries[prt];

    float u_C1, v_C1, Up;

    for(int j=0; j<boundary.number; j++)
    {
        int C0 = faces.connectivity[j].first, C1;
        switch(boundary.type(j))
        {
          case BoundaryType::VELOCITY:
            flowF.u(j) = boundary.conditions(j,0);
            flowF.v(j) = boundary.conditions(j,1);
            flowF.w(j) = boundary.conditions(j,2);
            flowF.p(j) = flowC.p(C0)+nabla.p.row(C0).dot(faces.d0.row(j));
            flowF.turb1(j) = boundary.conditions(j,4);
            flowF.turb2(j) = boundary.conditions(j,5);
            break;
          case BoundaryType::TOTALPRESSURE:
            assert(false); // NOT IMPLEMENTED YET
            break;
          case BoundaryType::PRESSURE:
            flowF.u(j) = flowC.u(C0);
            flowF.v(j) = flowC.v(C0);
            flowF.w(j) = flowC.w(C0);
            flowF.p(j) = boundary.conditions(j,3);
            flowF.turb1(j) = flowC.turb1(C0);
            flowF.turb2(j) = flowC.turb2(C0);
            break;
          case BoundaryType::MASSFLOW:
            // Same as pressure but needs iterative computation of static pressure
            assert(false); // NOT IMPLEMENTED YET
            break;
          case BoundaryType::WALL:
            flowF.u(j) = -boundary.conditions(j,0)*faces.centroid(j,1);
            flowF.v(j) =  boundary.conditions(j,0)*faces.centroid(j,0);
            flowF.w(j) = 0.0f;
            flowF.p(j) = flowC.p(C0)+nabla.p.row(C0).dot(faces.d0.row(j));

            Up = flowC.u(C0)*faces.area(j,0)+flowC.v(C0)*faces.area(j,1)+flowC.w(C0)*faces.area(j,2);
            Up = ((Matrix<float,1,3>() << flowC.u(C0), flowC.v(C0), flowC.w(C0)).finished()-
                  Up/faces.area.row(j).squaredNorm()*faces.area.row(j)).norm();
            boundary.conditions.block(j,1,1,5) = turbModel->wallValues(C0,Up,flowF.turb1(j),flowF.turb2(j));
            break;
          case BoundaryType::SYMMETRY:
            flowF.u(j) = flowC.u(C0)+limit.u(C0)*nabla.u.row(C0).dot(faces.d0.row(j));
            flowF.v(j) = flowC.v(C0)+limit.v(C0)*nabla.v.row(C0).dot(faces.d0.row(j));
            flowF.w(j) = flowC.w(C0)+limit.w(C0)*nabla.w.row(C0).dot(faces.d0.row(j));
            flowF.p(j) = flowC.p(C0)+nabla.p.row(C0).dot(faces.d0.row(j));
            flowF.turb1(j) = flowC.turb1(C0)+limit.turb1(C0)*nabla.turb1.row(C0).dot(faces.d0.row(j));
            flowF.turb2(j) = flowC.turb2(C0)+limit.turb2(C0)*nabla.turb2.row(C0).dot(faces.d0.row(j));
            break;
          case BoundaryType::PERIODIC:
            C1 = boundary.conditions(j,0); //Index of the neighbour cell
            // Velocity vector of neighbour cell rotated before interpolation
            u_C1 = flowC.u(C1)*boundary.conditions(j,2)-flowC.v(C1)*boundary.conditions(j,3),
            v_C1 = flowC.u(C1)*boundary.conditions(j,3)+flowC.v(C1)*boundary.conditions(j,2);
            flowF.u(j) = flowC.u(C0)*faces.wf(j)+u_C1*(1.0f-faces.wf(j));
            flowF.v(j) = flowC.v(C0)*faces.wf(j)+v_C1*(1.0f-faces.wf(j));
            flowF.w(j) = flowC.w(C0)*faces.wf(j)+flowC.w(C1)*(1.0f-faces.wf(j));
            flowF.p(j) = flowC.p(C0)*faces.wf(j)+flowC.p(C1)*(1.0f-faces.wf(j));
            flowF.turb1(j) = flowC.turb1(C0)*faces.wf(j)+flowC.turb1(C1)*(1.0f-faces.wf(j));
            flowF.turb2(j) = flowC.turb2(C0)*faces.wf(j)+flowC.turb2(C1)*(1.0f-faces.wf(j));
            break;
          case BoundaryType::GHOST:
            C1 = boundary.conditions(j,0); //Index of the ghost cell
            flowF.u(j) = flowC.u(C0)*faces.wf(j)+flowGC.u(C1)*(1.0f-faces.wf(j));
            flowF.v(j) = flowC.v(C0)*faces.wf(j)+flowGC.v(C1)*(1.0f-faces.wf(j));
            flowF.w(j) = flowC.w(C0)*faces.wf(j)+flowGC.w(C1)*(1.0f-faces.wf(j));
            flowF.p(j) = flowC.p(C0)*faces.wf(j)+flowGC.p(C1)*(1.0f-faces.wf(j));
            flowF.turb1(j) = flowC.turb1(C0)*faces.wf(j)+flowGC.turb1(C1)*(1.0f-faces.wf(j));
            flowF.turb2(j) = flowC.turb2(C0)*faces.wf(j)+flowGC.turb2(C1)*(1.0f-faces.wf(j));
            break;
        }
        // Reference frame change
        if(boundary.type(j)==BoundaryType::VELOCITY ||
           boundary.type(j)==BoundaryType::TOTALPRESSURE) {
            flowF.u(j) += m_rotationalSpeed*faces.centroid(j,1);
            flowF.v(j) -= m_rotationalSpeed*faces.centroid(j,0);
        }
    }
}
}
