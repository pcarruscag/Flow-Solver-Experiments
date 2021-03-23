//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"
#include "../../mesh/src/passageMesh.h"

namespace flow
{
int PressureBasedCoupledSolver::applyBndConds(const FlowParamManager& flowParams)
{
    #ifdef FLOW_VERBOSE_EXTRA
    std::cout << std::endl << "### Applying Boundary Conditions ###" << std::endl;
    #endif
    if(!flowParams.dataIsReady())
        return 1;
    if(!m_ctrlIsSet)
        return 2;
    if(!m_meshIsSet)
        return 3;

    m_bndrsSet = false;

    m_boundaries.resize(m_partNum);
    m_rotationalSpeed = flowParams.m_domainConditions.rotationalSpeed;
    m_rho = flowParams.m_fluidProperties.rho;
    m_mu  = flowParams.m_fluidProperties.mu;

    // Compute the area of inlet faces
    float velMagni, inletArea = 0.0;
    vector<float> inletAreas(m_partNum);
    #pragma omp parallel for reduction(+:inletArea)
    for(int i=0; i<m_partNum; i++)
        for(size_t j=0; j<(*m_part)[i].faces.groups.size(); j++)
            if((*m_part)[i].faces.groups[j].first == PassageMesh::INFLOW){
                inletAreas[i] = m_areaOfGroup(i,j);
                inletArea += inletAreas[i];
                break;
            }

    bool validOptions = true;
    #pragma omp parallel num_threads(m_partNum) private(velMagni) reduction(&&:validOptions)
    {
        int i = omp_get_thread_num();

        const mesh::UnstructuredMesh& mesh = (*m_part)[i];

        // Determine number of boundaries faces and resize data structures
        m_boundaries[i].number = 0;
        for(auto & group : mesh.faces.groups)
            m_boundaries[i].number += group.second.size();

        m_boundaries[i].type.resize(m_boundaries[i].number);
        m_boundaries[i].type.setZero();
        m_boundaries[i].conditions.resize(m_boundaries[i].number,6);
        m_boundaries[i].conditions.setZero();

        // Apply boundary conditions
        for(auto & group : mesh.faces.groups)
        {
            switch(group.first)
            {
              case PassageMesh::INFLOW:

                switch(flowParams.m_inletConditions.variable)
                {
                  case flowParams.VELOCITY:

                    switch(flowParams.m_inletConditions.direction)
                    {
                      case flowParams.NORMAL:

                        for(auto face : group.second)
                        {
                            float tmp = flowParams.m_inletConditions.scalar/mesh.faces.area.row(face).norm();
                            m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                            m_boundaries[i].conditions.block(face,0,1,3) = -tmp*mesh.faces.area.row(face);
                        }
                        break;

                      case flowParams.COMPONENTS:

                        if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                        {
                            for(auto face : group.second)
                            {
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions(face,0) = flowParams.m_inletConditions.components[0];
                                m_boundaries[i].conditions(face,1) = flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,2) = flowParams.m_inletConditions.components[2];
                            }
                        }else // CYLINDRICAL
                        {
                            for(auto face : group.second)
                            {
                                float invRadius = 1.0f/mesh.faces.centroid.block(face,0,1,2).norm();
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions(face,0) = invRadius*(
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,1) = invRadius*(
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,2) = flowParams.m_inletConditions.components[1];
                            }
                        }
                        break;

                      case flowParams.DIRECTION:

                        velMagni = flowParams.m_inletConditions.scalar/sqrt(
                                   pow(flowParams.m_inletConditions.components[0],2.0f)+
                                   pow(flowParams.m_inletConditions.components[1],2.0f)+
                                   pow(flowParams.m_inletConditions.components[2],2.0f));

                        if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                        {
                            for(auto face : group.second)
                            {
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions(face,0) = velMagni*flowParams.m_inletConditions.components[0];
                                m_boundaries[i].conditions(face,1) = velMagni*flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,2) = velMagni*flowParams.m_inletConditions.components[2];
                            }
                        }else // CYLINDRICAL
                        {
                            for(auto face : group.second)
                            {
                                float l = mesh.faces.centroid.block(face,0,1,2).norm();
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions(face,0) = velMagni/l*(
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,1) = velMagni/l*(
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,2) = velMagni*flowParams.m_inletConditions.components[1];
                            }
                        }
                        break;
                    }
                    break;

                  case flowParams.MASSFLOW:

                    switch(flowParams.m_inletConditions.direction)
                    {
                      case flowParams.NORMAL:

                        velMagni = flowParams.m_inletConditions.scalar/(m_rho*inletArea);
                        for(auto face : group.second)
                        {
                            float tmp = velMagni/mesh.faces.area.row(face).norm();
                            m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                            m_boundaries[i].conditions.block(face,0,1,3) = -tmp*mesh.faces.area.row(face);
                        }
                        break;

                      case flowParams.DIRECTION:

                        if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                        {
                            velMagni = 0.0;
                            for(auto face : group.second)
                            {
                                velMagni+= mesh.faces.area(face,0)*flowParams.m_inletConditions.components[0]+
                                           mesh.faces.area(face,1)*flowParams.m_inletConditions.components[1]+
                                           mesh.faces.area(face,2)*flowParams.m_inletConditions.components[2];
                            }
                            velMagni = flowParams.m_inletConditions.scalar*inletAreas[i]/(inletArea*m_rho*velMagni);
                            for(auto face : group.second)
                            {
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions(face,0) = velMagni*flowParams.m_inletConditions.components[0];
                                m_boundaries[i].conditions(face,1) = velMagni*flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,2) = velMagni*flowParams.m_inletConditions.components[2];
                            }
                        }else // CYLINDRICAL
                        {
                            velMagni = 0.0;
                            float dirNorm = 1.0/sqrt(
                                  pow(flowParams.m_inletConditions.components[0],2.0)+
                                  pow(flowParams.m_inletConditions.components[1],2.0)+
                                  pow(flowParams.m_inletConditions.components[2],2.0));
                            for(auto face : group.second)
                            {
                                float l = mesh.faces.centroid.block(face,0,1,2).norm(),
                                      vx = dirNorm/l*(
                                           mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                           mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[2]),
                                      vy = dirNorm/l*(
                                           mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                           mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[2]),
                                      vz = dirNorm*flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,0) = vx;
                                m_boundaries[i].conditions(face,1) = vy;
                                m_boundaries[i].conditions(face,2) = vz;
                                velMagni += mesh.faces.area(face,0)*vx+
                                            mesh.faces.area(face,1)*vy+
                                            mesh.faces.area(face,2)*vz;
                            }
                            velMagni = flowParams.m_inletConditions.scalar*inletAreas[i]/(inletArea*m_rho*velMagni);
                            for(auto face : group.second)
                            {
                                m_boundaries[i].type(face) = BoundaryType::VELOCITY;
                                m_boundaries[i].conditions.block(face,0,1,3) *= velMagni;
                            }
                        }
                        break;

                      case flowParams.COMPONENTS:
                        #pragma omp critical
                        std::cout << "Mass flow cannot be expressed in components.\n";
                        validOptions = false;
                    }
                    break;

                  case flowParams.TOTALPRESSURE:

                    switch(flowParams.m_inletConditions.direction)
                    {
                      case flowParams.NORMAL:

                        for(auto face : group.second)
                        {
                            float tmp = mesh.faces.area.row(face).norm();
                            m_boundaries[i].type(face) = BoundaryType::TOTALPRESSURE;
                            m_boundaries[i].conditions.block(face,0,1,3) = -tmp*mesh.faces.area.row(face);
                            m_boundaries[i].conditions(face,3) = flowParams.m_inletConditions.scalar;
                        }
                        break;

                      case flowParams.DIRECTION:
                      {
                        float dirNorm = 1.0/sqrt(
                              pow(flowParams.m_inletConditions.components[0],2.0)+
                              pow(flowParams.m_inletConditions.components[1],2.0)+
                              pow(flowParams.m_inletConditions.components[2],2.0));

                        if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                        {
                            for(auto face : group.second)
                            {
                                m_boundaries[i].type(face) = BoundaryType::TOTALPRESSURE;
                                m_boundaries[i].conditions(face,0) = dirNorm*flowParams.m_inletConditions.components[0];
                                m_boundaries[i].conditions(face,1) = dirNorm*flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,2) = dirNorm*flowParams.m_inletConditions.components[2];
                                m_boundaries[i].conditions(face,3) = flowParams.m_inletConditions.scalar;
                            }
                        }else // CYLINDRICAL
                        {
                            for(auto face : group.second)
                            {
                                float l = mesh.faces.centroid.block(face,0,1,2).norm();
                                m_boundaries[i].type(face) = BoundaryType::TOTALPRESSURE;
                                m_boundaries[i].conditions(face,0) = dirNorm/l*(
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,1) = dirNorm/l*(
                                    mesh.faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                    mesh.faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                                m_boundaries[i].conditions(face,2) = dirNorm*flowParams.m_inletConditions.components[1];
                                m_boundaries[i].conditions(face,3) = flowParams.m_inletConditions.scalar;
                            }
                        }
                        break;
                      }
                      case flowParams.COMPONENTS:
                        #pragma omp critical
                        std::cout << "Total pressure cannot be expressed in components.\n";
                        validOptions = false;
                    }
                    break;
                }
                if(flowParams.m_inletConditions.turbSpec == flowParams.KOMEGA)
                {
                    for(auto face : group.second)
                    {
                        m_boundaries[i].conditions(face,4) = flowParams.m_inletConditions.turbVal1;
                        m_boundaries[i].conditions(face,5) = flowParams.m_inletConditions.turbVal2;
                    }
                }else
                {
                    #pragma omp critical
                    std::cout << "Turbulence intensity specification not implemented.\n";
                    validOptions = false;
                }
                break;

              case PassageMesh::OUTFLOW:

                for(auto face : group.second)
                {
                    if(flowParams.m_outletConditions.massFlowOption){
                        m_boundaries[i].type(face) = BoundaryType::MASSFLOW;
                        m_boundaries[i].conditions(face,0) = flowParams.m_outletConditions.massFlow;
                    }else
                        m_boundaries[i].type(face) = BoundaryType::PRESSURE;
                    m_boundaries[i].conditions(face,3) = flowParams.m_outletConditions.pressure;
                }
                break;

              case PassageMesh::BLADE:

                for(auto face : group.second)
                {
                    m_boundaries[i].type(face) = BoundaryType::WALL;
                    m_boundaries[i].conditions(face,0) = 0.0f;
                }
                break;

              case PassageMesh::HUB:

                for(auto face : group.second)
                {
                    m_boundaries[i].type(face) = BoundaryType::WALL;
                    m_boundaries[i].conditions(face,0) = m_rotationalSpeed*
                        (float(flowParams.m_domainConditions.rotatingHub)-1.0f);
                }
                break;

              case PassageMesh::SHROUD:

                for(auto face : group.second)
                {
                    m_boundaries[i].type(face) = BoundaryType::WALL;
                    m_boundaries[i].conditions(face,0) = m_rotationalSpeed*
                        (float(flowParams.m_domainConditions.rotatingShroud)-1.0f);
                }
                break;

              case PassageMesh::PER_1:
              case PassageMesh::PER_2:
                for(auto face : group.second)
                    m_boundaries[i].type(face) = BoundaryType::PERIODIC;
                break;

              case UnstructuredMesh::GHOST:

                for(auto face : group.second)
                    m_boundaries[i].type(face) = BoundaryType::GHOST;
                break;
            }
        }
    }

    if(!validOptions) return 5;

    if(m_mapPeriodics()==0){
        if(m_mapGhostCells()==0){
            m_bndrsSet = true;
            return 0;
        }else
            return 4;
    }else
        return 3;
}
}
