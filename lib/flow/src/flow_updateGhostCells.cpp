//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
int PressureBasedCoupledSolver::m_updateGhostCells(const UpdateType type, const int partI)
{
    int partJ, startI, startJ, idxI, idxJ;
    if(type == FLOW_FIELD)
    {
        for(auto & comm : m_communications)
        {
            partJ = -1;
            if(partI==comm.partI) {partJ=comm.partJ; startI=comm.startI; startJ=comm.startJ;}
            if(partI==comm.partJ) {partJ=comm.partI; startI=comm.startJ; startJ=comm.startI;}
            if(partJ>-1)
              for(int i=0; i<comm.length; ++i) {
                idxI = startI++;
                idxJ = m_ghostCells[partJ].nghbrCell(startJ++);

                m_ghostCells[partI].flwFld.u(idxI) = m_flwFld_C[partJ].u(idxJ);
                m_ghostCells[partI].flwFld.v(idxI) = m_flwFld_C[partJ].v(idxJ);
                m_ghostCells[partI].flwFld.w(idxI) = m_flwFld_C[partJ].w(idxJ);
                m_ghostCells[partI].flwFld.p(idxI) = m_flwFld_C[partJ].p(idxJ);
                m_ghostCells[partI].flwFld.turb1(idxI) = m_flwFld_C[partJ].turb1(idxJ);
                m_ghostCells[partI].flwFld.turb2(idxI) = m_flwFld_C[partJ].turb2(idxJ);
            }
        }
    } else if(type == GRADS_LIMS)
    {
        for(auto & comm : m_communications)
        {
            partJ = -1;
            if(partI==comm.partI) {partJ=comm.partJ; startI=comm.startI; startJ=comm.startJ;}
            if(partI==comm.partJ) {partJ=comm.partI; startI=comm.startJ; startJ=comm.startI;}
            if(partJ>-1)
              for(int i=0; i<comm.length; ++i) {
                idxI = startI++;
                idxJ = m_ghostCells[partJ].nghbrCell(startJ++);

                m_ghostCells[partI].limit.u(idxI) = m_limit[partJ].u(idxJ);
                m_ghostCells[partI].limit.v(idxI) = m_limit[partJ].v(idxJ);
                m_ghostCells[partI].limit.w(idxI) = m_limit[partJ].w(idxJ);
                m_ghostCells[partI].limit.turb1(idxI) = m_limit[partJ].turb1(idxJ);
                m_ghostCells[partI].limit.turb2(idxI) = m_limit[partJ].turb2(idxJ);

                m_ghostCells[partI].mut(idxI) = m_mut[partJ](idxJ);

                m_ghostCells[partI].nabla.u.row(idxI) = m_nabla[partJ].u.row(idxJ);
                m_ghostCells[partI].nabla.v.row(idxI) = m_nabla[partJ].v.row(idxJ);
                m_ghostCells[partI].nabla.w.row(idxI) = m_nabla[partJ].w.row(idxJ);
                m_ghostCells[partI].nabla.p.row(idxI) = m_nabla[partJ].p.row(idxJ);
                m_ghostCells[partI].nabla.turb1.row(idxI) = m_nabla[partJ].turb1.row(idxJ);
                m_ghostCells[partI].nabla.turb2.row(idxI) = m_nabla[partJ].turb2.row(idxJ);
            }
        }
    } else if(type == DIAGONAL)
    {
        for(auto & comm : m_communications)
        {
            partJ = -1;
            if(partI==comm.partI) {partJ=comm.partJ; startI=comm.startI; startJ=comm.startJ;}
            if(partI==comm.partJ) {partJ=comm.partI; startI=comm.startJ; startJ=comm.startI;}
            if(partJ>-1)
              for(int i=0; i<comm.length; i++) {
                idxJ = m_ghostCells[partJ].nghbrCell(startJ++);
                m_ghostCells[partI].diag_nb(startI++) =
                  m_coefMat[partJ].getBlock(m_matDiagMap[partJ](idxJ)).diag(2);
            }
        }
    }
    return 0;
}
}
