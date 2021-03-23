//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

namespace flow
{
int PressureBasedCoupledSolver::m_mapGhostCells()
{
    #ifdef FLOW_VERBOSE_EXTRA
    std::cout << std::endl << "### Mapping Ghost Cells ###" << std::endl << std::endl;
    #endif
    bool matchFound;
    communication comm;
    vector<int> gcIdx(m_partNum,-1);
    vector<vector<int> > ghostFaces(m_partNum), faceIsMapped(m_partNum);
    m_ghostCells.resize(m_partNum);
    m_communications.resize(0);
    int Nc=0, Ngc=0;

    #pragma omp parallel num_threads(m_partNum) reduction(+:Nc,Ngc)
    {
        int prt = omp_get_thread_num();

        m_bndrsOfType(prt,BoundaryType::GHOST,ghostFaces[prt]);
        Ngc += m_ghostCells[prt].number = ghostFaces[prt].size();
        m_ghostCells[prt].nghbrPart.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].nghbrCell.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.u.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.v.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.w.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.p.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.turb1.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].flwFld.turb2.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].nabla.u.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].nabla.v.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].nabla.w.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].nabla.p.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].nabla.turb1.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].nabla.turb2.resize(m_ghostCells[prt].number,NoChange);
        m_ghostCells[prt].limit.u.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].limit.v.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].limit.w.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].limit.turb1.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].limit.turb2.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].coefMat.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].coefMat_t.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].vol_nb.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].diag_nb.resize(m_ghostCells[prt].number);
        m_ghostCells[prt].mut.resize(m_ghostCells[prt].number);
        faceIsMapped[prt].resize(m_ghostCells[prt].number,0);
        Nc += (*m_part)[prt].cells.number;
    }
    #ifdef FLOW_VERBOSE_EXTRA
    std::cout << "  Average Overlap: " << float(100*Ngc)/Nc << "%" << std::endl << std::endl;
    #endif

    for(int prtI=0; prtI<m_partNum-1; prtI++)
    {
        for(int prtJ=prtI+1; prtJ<m_partNum; prtJ++)
        {
            matchFound = false;
            int cursorI = -1;
            for(auto & fI : ghostFaces[prtI])
            {
              if(faceIsMapped[prtI][++cursorI]==0)
              {
                float characLen = std::sqrt((*m_part)[prtI].faces.area.row(fI).norm());
                int cursorJ = -1;
                for(auto & fJ : ghostFaces[prtJ])
                {
                  if(faceIsMapped[prtJ][++cursorJ]==0)
                  {
                    if((((*m_part)[prtI].faces.centroid.row(fI)-
                         (*m_part)[prtJ].faces.centroid.row(fJ)).norm() < m_control.mapTol*characLen) &&
                       (((*m_part)[prtI].faces.area.row(fI)+
                         (*m_part)[prtJ].faces.area.row(fJ)).norm() < m_control.mapTol*characLen*characLen))
                    {
                        faceIsMapped[prtI][cursorI] = faceIsMapped[prtJ][cursorJ] = 1;

                        gcIdx[prtI]++;
                        m_boundaries[prtI].conditions(fI,0) = gcIdx[prtI];
                        m_ghostCells[prtI].nghbrPart(gcIdx[prtI]) = prtJ;
                        m_ghostCells[prtI].nghbrCell(gcIdx[prtI]) = (*m_part)[prtI].faces.connectivity[fI].first;
                        m_ghostCells[prtI].coefMat[gcIdx[prtI]].setZero(4,4);
                        m_ghostCells[prtI].coefMat_t[gcIdx[prtI]].setZero(1,1);
                        m_ghostCells[prtI].vol_nb(gcIdx[prtI]) =
                            (*m_part)[prtJ].cells.volume((*m_part)[prtJ].faces.connectivity[fJ].first);

                        gcIdx[prtJ]++;
                        m_boundaries[prtJ].conditions(fJ,0) = gcIdx[prtJ];
                        m_ghostCells[prtJ].nghbrPart(gcIdx[prtJ]) = prtI;
                        m_ghostCells[prtJ].nghbrCell(gcIdx[prtJ]) = (*m_part)[prtJ].faces.connectivity[fJ].first;
                        m_ghostCells[prtJ].coefMat[gcIdx[prtJ]].setZero(4,4);
                        m_ghostCells[prtJ].coefMat_t[gcIdx[prtJ]].setZero(1,1);
                        m_ghostCells[prtJ].vol_nb(gcIdx[prtJ]) =
                            (*m_part)[prtI].cells.volume((*m_part)[prtI].faces.connectivity[fI].first);

                        (*m_part)[prtI].faces.alfa1(fI) = -(*m_part)[prtJ].faces.alfa0(fJ);
                        (*m_part)[prtI].faces.r1.row(fI) = (*m_part)[prtJ].faces.r0.row(fJ);
                        (*m_part)[prtI].faces.d1.row(fI) = (*m_part)[prtJ].faces.d0.row(fJ);
                        (*m_part)[prtI].faces.wf(fI) = 1.0f/(1.0f+(*m_part)[prtI].faces.r0.row(fI).norm()/
                                                                  (*m_part)[prtI].faces.r1.row(fI).norm());

                        (*m_part)[prtJ].faces.alfa1(fJ) = -(*m_part)[prtI].faces.alfa0(fI);
                        (*m_part)[prtJ].faces.r1.row(fJ) = (*m_part)[prtI].faces.r0.row(fI);
                        (*m_part)[prtJ].faces.d1.row(fJ) = (*m_part)[prtI].faces.d0.row(fI);
                        (*m_part)[prtJ].faces.wf(fJ) = 1.f-(*m_part)[prtI].faces.wf(fI);

                        // Register the connection between partitions
                        if(!matchFound)
                        {
                            comm.partI = prtI;
                            comm.partJ = prtJ;
                            comm.startI = gcIdx[prtI];
                            comm.startJ = gcIdx[prtJ];
                            matchFound = true;
                        }
                        break;
                    }
                  }
                }
              }
            }
            if(matchFound)
            {
                comm.length = gcIdx[prtI]-comm.startI+1;
                m_communications.push_back(comm);
                #ifdef FLOW_VERBOSE_EXTRA
                std::cout << "  " << comm.length << " connections between parts "
                          << comm.partI+1 << " and " << comm.partJ+1 << std::endl;
                #endif
            }
        }
    }
    // make sure all ghost cells were mapped
    for(int prt=0; prt<m_partNum; prt++)
        assert(gcIdx[prt]+1==m_ghostCells[prt].number);
    return 0;
}
}
