//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
void PressureBasedCoupledSolver::m_computeLimiters(const int prt,
                                                   const VectorXf& phi,
                                                   const VectorXf& phiGC,
                                                   const VectorXf& phiF,
                                                   const MatrixX3f& nablaPhi,
                                                   VectorXf& limit) const
{
    auto& msh = (*m_part)[prt];

    limit.setConstant(2.0);
    VectorXf maxNghbr(msh.cells.number), minNghbr(msh.cells.number);
    maxNghbr.setZero();
    minNghbr.setZero();

    auto venkatFcn = [](float x, float y, float eps){
        return ((x*x+eps)*y+2.0f*y*y*x)/(y*(x*x+2.0f*y*y+y*x+eps));
    };

    int bndNum = m_boundaries[prt].number;

    for(int i=0; i<bndNum; i++)
    {
        int C0 = msh.faces.connectivity[i].first; float phiC1;

        if(m_boundaries[prt].type(i)==BoundaryType::GHOST)
            phiC1 = phiGC(int(m_boundaries[prt].conditions(i,0)));
        else
            phiC1 = phiF(i);

        maxNghbr(C0) = max(maxNghbr(C0),phiC1-phi(C0));
        minNghbr(C0) = min(minNghbr(C0),phiC1-phi(C0));
    }
    for(int i=bndNum; i<msh.faces.number; i++)
    {
        int C0 = msh.faces.connectivity[i].first,
            C1 = msh.faces.connectivity[i].second;
        maxNghbr(C0) = max(maxNghbr(C0),phi(C1)-phi(C0));
        minNghbr(C0) = min(minNghbr(C0),phi(C1)-phi(C0));
        maxNghbr(C1) = max(maxNghbr(C1),phi(C0)-phi(C1));
        minNghbr(C1) = min(minNghbr(C1),phi(C0)-phi(C1));
    }
    for(int i=0; i<msh.faces.number; i++)
    {
        int C0 = msh.faces.connectivity[i].first;
        float deltaPhi = nablaPhi.row(C0).dot(msh.faces.r0.row(i)),
              eps = m_control.venkatK*msh.cells.volume(C0);
        if(deltaPhi > 0.0f)
            limit(C0) = min(limit(C0),venkatFcn(maxNghbr(C0),deltaPhi,eps));
        else if(deltaPhi < 0.0f)
            limit(C0) = min(limit(C0),venkatFcn(minNghbr(C0),deltaPhi,eps));
        else
            limit(C0) = min(limit(C0),1.0f);
    }
    for(int i=bndNum; i<msh.faces.number; i++)
    {
        int C1 = msh.faces.connectivity[i].second;
        float deltaPhi = nablaPhi.row(C1).dot(msh.faces.r1.row(i)),
              eps = m_control.venkatK*msh.cells.volume(C1);
        if(deltaPhi > 0.0f)
            limit(C1) = min(limit(C1),venkatFcn(maxNghbr(C1),deltaPhi,eps));
        else if(deltaPhi < 0.0f)
            limit(C1) = min(limit(C1),venkatFcn(minNghbr(C1),deltaPhi,eps));
        else
            limit(C1) = min(limit(C1),1.0f);
    }
}
}
