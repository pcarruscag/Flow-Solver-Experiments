//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

#include <cmath>

namespace flow
{
int PressureBasedCoupledSolver::m_mapPeriodics()
{
    #ifdef FLOW_VERBOSE_EXTRA
    std::cout << std::endl << "### Mapping Periodic Boundaries ###" << std::endl;
    #endif
    #pragma omp parallel num_threads(m_partNum)
    {
        const int i = omp_get_thread_num();
        auto& faces = (*m_part)[i].faces;

        std::vector<bool> mapped(m_boundaries[i].number,false);
        float r1, r2, z1, z2, characLen;
        for(int j=0; j<m_boundaries[i].number; j++)
        {
            if(m_boundaries[i].type(j)==BoundaryType::PERIODIC && !mapped[j])
            {
                r1 = std::sqrt(std::pow(faces.centroid(j,0),2.0)+std::pow(faces.centroid(j,1),2.0));
                z1 = faces.centroid(j,2);
                characLen = std::sqrt(faces.area.row(j).norm());
                int k;
                for(k=j+1; k<m_boundaries[i].number; k++)
                {
                    if(m_boundaries[i].type(k)==BoundaryType::PERIODIC)
                    {
                        r2 = std::sqrt(std::pow(faces.centroid(k,0),2.0)+std::pow(faces.centroid(k,1),2.0));
                        z2 = faces.centroid(k,2);
                        if((std::abs(r1-r2)+std::abs(z1-z2))<m_control.mapTol*characLen) break; // k matches j
                    }
                }
                assert(k<m_boundaries[i].number); // a match was not found

                faces.alfa1(j) = -faces.alfa0(k);
                // determine the rotation angle
                float theta = acos((faces.centroid(j,0)*faces.centroid(k,0)+faces.centroid(j,1)*faces.centroid(k,1))/
                                   (faces.centroid.block(j,0,1,2).norm()*faces.centroid.block(k,0,1,2).norm())),
                      delta = faces.centroid(j,0)*faces.centroid(k,1)-faces.centroid(j,1)*faces.centroid(k,0);
                if(delta > 0)
                    theta *= -1.0f;

                faces.r1.row(j) = faces.r0.row(k);
                faces.d1.row(j) = faces.d0.row(k);
                faces.wf(j) = 1.0f/(1.0f+faces.r0.row(j).norm()/faces.r1.row(j).norm());

                // Face j is made master
                m_boundaries[i].conditions(j,0) = faces.connectivity[k].first;
                m_boundaries[i].conditions(j,1) = 1;
                m_boundaries[i].conditions(j,2) = cos(theta);
                m_boundaries[i].conditions(j,3) = sin(theta);
                m_boundaries[i].conditions(j,4) = k;

                // Face k is made "do nothing", wf is still needed for interpolation
                m_boundaries[i].conditions(k,0) = faces.connectivity[j].first;
                m_boundaries[i].conditions(k,1) = 0;
                m_boundaries[i].conditions(k,2) = cos(theta);
                m_boundaries[i].conditions(k,3) = -1.0*sin(theta); // opposite direction for rotation
                m_boundaries[i].conditions(k,4) = j;
                faces.wf(k) = 1.0f-faces.wf(j);
                mapped[k] = true;
            }
        }
    }
    return 0;
}
}
