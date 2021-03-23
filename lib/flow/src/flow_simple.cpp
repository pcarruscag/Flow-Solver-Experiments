//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow_simple.h"

#include "../../mathUtils/src/matrix.h"
#include "../../mathUtils/src/preconditioners.h"
#include "../../mathUtils/src/solvers.h"

using namespace mathUtils;

namespace flow
{
PressureBasedSegregatedSolver::PressureBasedSegregatedSolver() :
    m_gradCalcX(mappedCSRf_t(0,0,0,0,0,0)), m_gradCalcY(mappedCSRf_t(0,0,0,0,0,0)),
    m_gradCalcZ(mappedCSRf_t(0,0,0,0,0,0)), m_sumFluxes(mappedCSRf_t(0,0,0,0,0,0))
{
    m_ctrlIsSet = false;
    m_meshIsSet = false;
    m_bndrsSet  = false;
    m_isInit    = false;
    m_finished  = false;
}

void PressureBasedSegregatedSolver::setControl(const FlowParamManager& flowParams)
{
    m_control = flowParams.m_controlParams;
    m_ctrlIsSet = true;
}

int PressureBasedSegregatedSolver::setMesh(UnstructuredMesh* mesh)
{
    m_meshIsSet = false;
    if(mesh->status()!=0) return 1;
    m_mesh = mesh;
    m_meshIsSet = true;
    return 0;
}

int PressureBasedSegregatedSolver::applyBndConds(const FlowParamManager& flowParams)
{
    #ifdef FLOW_VERBOSE_EXTRA
    std::cout << std::endl << "### Applying Boundary Conditions ###" << std::endl;
    #endif
    if(!flowParams.dataIsReady()) return 1;
    if(!m_ctrlIsSet) return 2;
    if(!m_meshIsSet) return 3;

    m_bndrsSet = false;

    m_rho = flowParams.m_fluidProperties.rho;
    m_mu  = flowParams.m_fluidProperties.mu;

    // Determine number of boundaries faces and resize data structures
    m_boundaries.number = 0;
    for(size_t j=0; j<m_mesh->faces.groups.size(); j++)
        m_boundaries.number += m_mesh->faces.groups[j].second.size();

    m_boundaries.type.setZero(m_boundaries.number);
    m_boundaries.conditions.setZero(m_boundaries.number,6);

    // Apply boundary conditions
    for(auto & group : m_mesh->faces.groups)
    {
        float_t velMagni;

        switch(group.first)
        {
          case 2: //Inlet

            switch(flowParams.m_inletConditions.variable)
            {
              case flowParams.VELOCITY:

                switch(flowParams.m_inletConditions.direction)
                {
                  case flowParams.NORMAL:

                    for(auto face : group.second)
                    {
                        float_t tmp = flowParams.m_inletConditions.scalar/
                                      m_mesh->faces.area.row(face).norm();
                        m_boundaries.type(face) = 1;
                        m_boundaries.conditions.block(face,0,1,3) = -tmp*m_mesh->faces.area.row(face);
                    }
                    break;

                  case flowParams.COMPONENTS:

                    if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                    {
                        for(auto face : group.second)
                        {
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = flowParams.m_inletConditions.components[0];
                            m_boundaries.conditions(face,1) = flowParams.m_inletConditions.components[1];
                            m_boundaries.conditions(face,2) = flowParams.m_inletConditions.components[2];
                        }
                    }else // CYLINDRICAL
                    {
                        for(auto face : group.second)
                        {
                            float_t invRadius = 1.0f/m_mesh->faces.centroid.block(face,0,1,2).norm();
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = invRadius*(
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,1) = invRadius*(
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,2) = flowParams.m_inletConditions.components[1];
                        }
                    }
                    break;

                  case flowParams.DIRECTION:

                    velMagni = flowParams.m_inletConditions.scalar/float_t(sqrt(
                               pow(flowParams.m_inletConditions.components[0],2.0f)+
                               pow(flowParams.m_inletConditions.components[1],2.0f)+
                               pow(flowParams.m_inletConditions.components[2],2.0f)));

                    if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                    {
                        for(auto face : group.second)
                        {
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = velMagni*flowParams.m_inletConditions.components[0];
                            m_boundaries.conditions(face,1) = velMagni*flowParams.m_inletConditions.components[1];
                            m_boundaries.conditions(face,2) = velMagni*flowParams.m_inletConditions.components[2];
                        }
                    }else // CYLINDRICAL
                    {
                        for(auto face : group.second)
                        {
                            float_t l = m_mesh->faces.centroid.block(face,0,1,2).norm();
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = velMagni/l*(
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,1) = velMagni/l*(
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,2) = velMagni*flowParams.m_inletConditions.components[1];
                        }
                    }
                    break;
                }
                break;

              default: assert(false && "Inlet type not implemented");
            }
            break;
          case 3: //Outlet
            for(auto face : group.second)
            {
                m_boundaries.type(face) = 3;
                m_boundaries.conditions(face,3) = flowParams.m_outletConditions.pressure;
            }
            break;
          case 6: //Blade
            for(auto face : group.second)
                m_boundaries.type(face) = 5;
            break;
          case 0: //Bottom
            for(auto face : group.second)
                m_boundaries.type(face) = 5;
            break;
          case 1: //Top
            for(auto face : group.second)
                m_boundaries.type(face) = 5;
            break;
          case 4: //Periodic 1
            for(auto face : group.second)
                m_boundaries.type(face) = 5;
            break;
          case 5: //Periodic 2
            for(auto face : group.second)
                m_boundaries.type(face) = 5;
            break;
        }
    }
    m_bndrsSet = true;
    return 0;
}


int PressureBasedSegregatedSolver::run()
{
    /*
    Cell-based connectivity is used to enable shared-memory parallelism.

    Most steps of the solver will be expressed as matrix vector products, for now
    those matrices will be in CSR format, but to run on GPU's efficiently the list
    of lists (LIL) format will have to be used.

    Pipeline of operations inside the main loop:

    1 - Face values (u,v,w,p)
        This will be expressed as u_f = A*u_c+u_f0
        "f" for faces, "c" for cells, A is interpolation matrix, and u_f0 the
        vector of given values (boundary conditions and grid velocities).
        The interpolation matrix could also handle the projection at boundaries,
        however that would require one matrix for velocities and another for
        pressure, so A will take care only of internal faces. The project value will
        be expressed some other way and put in u_f0.
    2 - Gradients (u,v,w,p)
        Matrix vector product with the face values, the matrix has outward pointing
        areas for each cell. It may be worth breaking down into xyz as that would
        allow reuse when computing the mass imbalance, this also saves memory by
        reusing indexes (which are just the face adjacency!)
    3 - Limiters (u,v,w)
        This requires min/max operations and projections to all faces, using matrix-
        -vector products would use too much memory. For each cell we get the min and
        neighbour and associated projection. This will probably be the least GPU
        friendly step.
    4 - Momentum (u,v,w) source term
        This encompasses the convective and diffusive corrections and the pressure
        gradient. The pressure gradient is already computed by this stage, the
        corrections are first computed for the faces and then summed (integrated)
        to obtain the cell value. The summation is an Mv product which could be re-
        used to compute mass imbalances (and also the gradients?)
    5 - Momentum (u,v,w) matrix
        Compute the diffusion coefficient for all faces (this can be done in step 4
        and used here), it contributes to cells C0 and C1. Then add max(-mdot,0) to
        this quantity and set the anb0's, then add mdot to it and set the anb1's.
        Actually, the formula for the anb0/1's is Df+max(-/+mdot,0), they are the
        same except for the sign. We can loop through the nz's of the matrix, for
        each we know the face and we can infer C0 and C1 from the direction matrix.
        This leaves out the center coefficient and so this process would only be
        efficiency for the LIL format.
    X - Other SIMPLE steps
        The other steps of the algorithm are similar to the previous ones and
        warrant no in-depth discussion.
    */

    using std::cout;

    if(!m_meshIsSet) return 1;
    if(!m_bndrsSet) return 2;

    cout << std::scientific;
    cout.precision(3);
    cout << "\n\n";
    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "|                                                                                         |\n";
    cout << "|                                     SOLVER  STARTED                                     |\n";
    cout << "|                                                                                         |\n";
    cout << "-------------------------------------------------------------------------------------------\n";

    cout << std::left << std::setw(90) << "| Allocating memory for run..." << "|\n";

    m_finished = false;

    m_allocate();
    m_buildAdjacency();
    m_buildFaceIterp();
    m_buildGradCalc();
    m_buildMatAssembler();

    if(!m_isInit) m_defaultInit();

    // prepare linear solver
    solvers::BiCGSTAB<SparseMatrix<float_t,RowMajor>,matrixX3f_t> solverV;
    preconditioners::ILU0<matrixX3f_t> precV;

    solverV.setData(m_systemMat, m_momentumSrc, m_momentumSol);
    solverV.setPreconditioner(&precV);

    solvers::ConjugateGradient<SparseMatrix<float_t,RowMajor>,vectorXf_t> solverP;
    preconditioners::ILU0<vectorXf_t> precP;

    solverP.setData(m_systemMat, m_pcorrectSrc, m_pcorrectSol);
    solverP.setPreconditioner(&precP);

    Array<float_t,1,4> res, resNorm; res.setOnes(); resNorm.setOnes();

    double ttot = -omp_get_wtime();
    double tlin1 = 0.0, tlin2 = 0.0;

    // outer iterations
    for(int iter=1; iter<=m_control.maxIt; iter++)
    {
        // relaxation factors
        float_t alphaP, alphaV = 0.01f*m_control.cflNumber;

        if(iter < m_control.minIt/2) {
            alphaP = m_control.relax0;
        } else if(iter < m_control.minIt) {
            // ramp relaxation factors
            float_t blend = 2.0f*float_t(iter)/float_t(m_control.minIt)-1.0f;
            alphaP = blend*m_control.relax1+(1.0f-blend)*m_control.relax0;
        } else {
            alphaP = m_control.relax1;
        }

        // face values
        m_boundaryValues();
        m_flwFld_F.u = m_faceInterpMat*m_flwFld_C.u+m_faceGivenVals.u;
        m_flwFld_F.v = m_faceInterpMat*m_flwFld_C.v+m_faceGivenVals.v;
        m_flwFld_F.w = m_faceInterpMat*m_flwFld_C.w+m_faceGivenVals.w;
        m_flwFld_F.p = m_faceInterpMat*m_flwFld_C.p+m_faceGivenVals.p;

        // gradients
        #define GRADS(phi)\
        m_nabla.phi.col(0) = (m_gradCalcX*m_flwFld_F.phi).cwiseQuotient(m_mesh->cells.volume);\
        m_nabla.phi.col(1) = (m_gradCalcY*m_flwFld_F.phi).cwiseQuotient(m_mesh->cells.volume);\
        m_nabla.phi.col(2) = (m_gradCalcZ*m_flwFld_F.phi).cwiseQuotient(m_mesh->cells.volume)
        GRADS(u); GRADS(v); GRADS(w); GRADS(p);
        #undef GRADS

        // limiters
        m_computeLimiters(m_flwFld_C.u.data(),m_flwFld_F.u.data(),&m_nabla.u,m_limit.u.data());
        m_computeLimiters(m_flwFld_C.v.data(),m_flwFld_F.v.data(),&m_nabla.v,m_limit.v.data());
        m_computeLimiters(m_flwFld_C.w.data(),m_flwFld_F.w.data(),&m_nabla.w,m_limit.w.data());

        // diffusion coefficients
        for(int i=0; i<m_mesh->faces.number; ++i)
            m_diffCoeff(i) = m_mu/(m_mesh->faces.alfa0(i)-m_mesh->faces.alfa1(i));

        // clear diagonal of momentum matrix (or maybe clear everything?)
        for(int i=0; i<m_mesh->cells.number; ++i)
            m_matAssembler.nzPtr[m_matAssembler.diagonal[i]] = 0.0;

        // momentum correction fluxes (2nd order) (some boundaries add to the diagonal)
        m_compute2ndOrderFluxes();

        // momentum matrix
        // set anb0's
        m_diffCoeff -= m_mdot.cwiseMin(0.0);
        for(int i=0; i<m_matAssembler.Nnz0; ++i)
          m_matAssembler.nzPtr[m_matAssembler.nz0pos[i]]=-m_diffCoeff(m_matAssembler.nz0face[i]);

        // set anb1's
        m_diffCoeff += m_mdot;
        for(int i=0; i<m_matAssembler.Nnz1; ++i)
          m_matAssembler.nzPtr[m_matAssembler.nz1pos[i]]=-m_diffCoeff(m_matAssembler.nz1face[i]);

        // set diagonal
        for(int i=0; i<m_mesh->cells.number; ++i)
        {
            float_t tmp = -m_systemMat.row(i).sum()/alphaV;
            m_matAssembler.nzPtr[m_matAssembler.diagonal[i]] = tmp;
            m_matAssembler.momDiagCoeff(i) = tmp;

            m_momentumSrc(i,0) = (1.0f-alphaV)*tmp*m_flwFld_C.u(i);
            m_momentumSrc(i,1) = (1.0f-alphaV)*tmp*m_flwFld_C.v(i);
            m_momentumSrc(i,2) = (1.0f-alphaV)*tmp*m_flwFld_C.w(i);
        }

        // momentum source terms
        m_momentumSrc += m_sumFluxes*m_2ndOrderFluxes;
        for(int i=0; i<m_mesh->cells.number; ++i)
          m_momentumSrc.row(i) -= m_nabla.p.row(i)*m_mesh->cells.volume(i);

        // solve the momentum equations (in residual form)
        tlin1 -= omp_get_wtime();

        m_momentumSrc.col(0) -= m_systemMat*m_flwFld_C.u;
        m_momentumSrc.col(1) -= m_systemMat*m_flwFld_C.v;
        m_momentumSrc.col(2) -= m_systemMat*m_flwFld_C.w;

        precV.compute(m_systemMat);

        m_momentumSol.setZero();
        solverV.solve(m_control.linSolMaxIt, m_control.linSolTol);
        m_flwFld_C.u += m_momentumSol.col(0);
        m_flwFld_C.v += m_momentumSol.col(1);
        m_flwFld_C.w += m_momentumSol.col(2);

        tlin1 += omp_get_wtime();

        // pressure correction matrix
        // clear diagonal of matrix
        for(int i=0; i<m_mesh->cells.number; ++i)
            m_matAssembler.nzPtr[m_matAssembler.diagonal[i]] = 0.0;

        // "diffusion" coefficient and pressure outlet boundaries
        for(int i=0; i<m_boundaries.number; ++i)
        {
            int C0 = m_mesh->faces.connectivity[i].first;

            m_diffCoeff(i) = m_rho*m_mesh->cells.volume(C0)/
                (m_matAssembler.momDiagCoeff(C0)*m_mesh->faces.alfa0(i));

            if(m_boundaries.type(i)==3)
                m_matAssembler.nzPtr[m_matAssembler.diagonal[C0]] += m_diffCoeff(i);
        }

        for(int i=m_boundaries.number; i<m_mesh->faces.number; ++i)
        {
            int C0 = m_mesh->faces.connectivity[i].first,
                C1 = m_mesh->faces.connectivity[i].second;

            m_diffCoeff(i) = m_rho*(m_mesh->cells.volume(C0)+m_mesh->cells.volume(C1))/
                (m_matAssembler.momDiagCoeff(C0)+m_matAssembler.momDiagCoeff(C1))/
                (m_mesh->faces.alfa0(i)-m_mesh->faces.alfa1(i));
        }
        // set anb's
        for(int i=0; i<m_matAssembler.Nnz0; ++i)
          m_matAssembler.nzPtr[m_matAssembler.nz0pos[i]]=m_diffCoeff(m_matAssembler.nz0face[i]);

        for(int i=0; i<m_matAssembler.Nnz1; ++i)
          m_matAssembler.nzPtr[m_matAssembler.nz1pos[i]]=m_diffCoeff(m_matAssembler.nz1face[i]);

        // set diagonal
        for(int i=0; i<m_mesh->cells.number; ++i)
        {
            float_t tmp = -m_systemMat.row(i).sum();
            m_matAssembler.nzPtr[m_matAssembler.diagonal[i]] = tmp;
        }

        // compute the momentum interpolated face fluxes
        // velocity contribution (linear interpolation)
        m_boundaryValues();
        m_flwFld_F.u = m_faceInterpMat*m_flwFld_C.u+m_faceGivenVals.u;
        m_flwFld_F.v = m_faceInterpMat*m_flwFld_C.v+m_faceGivenVals.v;
        m_flwFld_F.w = m_faceInterpMat*m_flwFld_C.w+m_faceGivenVals.w;

        m_mdot = m_rho*(m_mesh->faces.area.col(0).cwiseProduct(m_flwFld_F.u)+
                        m_mesh->faces.area.col(1).cwiseProduct(m_flwFld_F.v)+
                        m_mesh->faces.area.col(2).cwiseProduct(m_flwFld_F.w));

        // added dissipation term
        for(int i=0; i<m_boundaries.number; ++i)
        {
            if(m_boundaries.type(i)==3)
            {
                int C0 = m_mesh->faces.connectivity[i].first;

                m_mdot(i) -= m_diffCoeff(i)*(m_flwFld_F.p(i)-m_flwFld_C.p(C0)-
                             m_nabla.p.row(C0).dot(m_mesh->faces.r0.row(i)));
            }
        }
        for(int i=m_boundaries.number; i<m_mesh->faces.number; ++i)
        {
            int C0 = m_mesh->faces.connectivity[i].first,
                C1 = m_mesh->faces.connectivity[i].second;

            float_t wf = m_mesh->faces.wf(i);
            Matrix<float_t,1,3> nablaPf = wf*m_nabla.p.row(C0)+(1.0f-wf)*m_nabla.p.row(C1);

            m_mdot(i) += m_diffCoeff(i)*(m_flwFld_C.p(C0)+nablaPf.dot(m_mesh->faces.r0.row(i))-
                                         m_flwFld_C.p(C1)-nablaPf.dot(m_mesh->faces.r1.row(i)));
        }

        // pressure correction source term (mass imbalances)
        m_pcorrectSrc = m_sumFluxes*m_mdot;

        // solve for the pressure corrections
        tlin2 -= omp_get_wtime();
        precP.compute(m_systemMat);
        m_pcorrectSol.setZero();
        solverP.solve(m_control.linSolMaxIt, m_control.linSolTol);
        tlin2 += omp_get_wtime();

        // correct face fluxes (no relaxation)
        for(int i=0; i<m_boundaries.number; ++i)
          if(m_boundaries.type(i)==3)
            m_mdot(i) += m_diffCoeff(i)*m_pcorrectSol(m_mesh->faces.connectivity[i].first);

        for(int i=m_boundaries.number; i<m_mesh->faces.number; ++i)
        {
            int C0 = m_mesh->faces.connectivity[i].first,
                C1 = m_mesh->faces.connectivity[i].second;

            m_mdot(i) -= m_diffCoeff(i)*(m_pcorrectSol(C1)-m_pcorrectSol(C0));
        }

        // correct the velocities (no relaxation)
        m_flwFld_F.p = m_faceInterpMat*m_pcorrectSol;
        for(int i=0; i<m_boundaries.number; ++i)
          if(m_boundaries.type(i)==1 || m_boundaries.type(i)==5)
            m_flwFld_F.p(i) = m_pcorrectSol(m_mesh->faces.connectivity[i].first);

        m_flwFld_C.u -= (m_gradCalcX*m_flwFld_F.p).cwiseQuotient(m_matAssembler.momDiagCoeff);
        m_flwFld_C.v -= (m_gradCalcY*m_flwFld_F.p).cwiseQuotient(m_matAssembler.momDiagCoeff);
        m_flwFld_C.w -= (m_gradCalcZ*m_flwFld_F.p).cwiseQuotient(m_matAssembler.momDiagCoeff);

        // update pressure (with relaxation)
        m_flwFld_C.p += alphaP*m_pcorrectSol;

        // monitor convergence
        for(int i=0; i<3; ++i) res(i) = m_momentumSrc.col(i).norm();
        res(3) = m_pcorrectSrc.norm();

        if(iter<3)
          for(int i=0; i<4; ++i)
            if((resNorm(i)==1.0f && res(i)!=0.0f) || res(i)>5.0f*resNorm(i))
              resNorm(i) = res(i);
        cout << iter << " - " << res/resNorm << std::endl;

        if((res/resNorm < m_control.tol).all()) break;
    }

    ttot += omp_get_wtime();

    cout << ttot-tlin1-tlin2 << "  " << tlin1 << "  " << tlin2 << "  " << ttot << std::endl;

    // update face values before post-processing
    m_boundaryValues();
    m_flwFld_F.u = m_faceInterpMat*m_flwFld_C.u+m_faceGivenVals.u;
    m_flwFld_F.v = m_faceInterpMat*m_flwFld_C.v+m_faceGivenVals.v;
    m_flwFld_F.w = m_faceInterpMat*m_flwFld_C.w+m_faceGivenVals.w;
    m_flwFld_F.p = m_faceInterpMat*m_flwFld_C.p+m_faceGivenVals.p;

    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "|                                                                                         |\n";
    cout << "|                                     SOLVER FINISHED                                     |\n";
    cout << "|                                                                                         |\n";
    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "\n\n";

    return 0;
}


int PressureBasedSegregatedSolver::m_allocate()
{
    int nCells = m_mesh->cells.number,
        nFaces = m_mesh->faces.number;

    m_flwFld_C.u.resize(nCells);
    m_flwFld_C.v.resize(nCells);
    m_flwFld_C.w.resize(nCells);
    m_flwFld_C.p.resize(nCells);

    m_flwFld_F.u.resize(nFaces);
    m_flwFld_F.v.resize(nFaces);
    m_flwFld_F.w.resize(nFaces);
    m_flwFld_F.p.resize(nFaces);

    m_faceGivenVals.u.setZero(nFaces);
    m_faceGivenVals.v.setZero(nFaces);
    m_faceGivenVals.w.setZero(nFaces);
    m_faceGivenVals.p.setZero(nFaces);

    m_nabla.u.setZero(nCells,3);
    m_nabla.v.setZero(nCells,3);
    m_nabla.w.setZero(nCells,3);
    m_nabla.p.setZero(nCells,3);

    m_limit.u.setZero(nCells);
    m_limit.v.setZero(nCells);
    m_limit.w.setZero(nCells);

    m_momentumSrc.resize(nCells,3);
    m_momentumSol.resize(nCells,3);
    m_pcorrectSrc.resize(nCells);
    m_pcorrectSol.resize(nCells);

    m_mdot.resize(nFaces);
    m_diffCoeff.resize(nFaces);
    m_2ndOrderFluxes.resize(nFaces,3);

    return 0;
}


int PressureBasedSegregatedSolver::m_buildAdjacency()
{
    int Nf = m_mesh->faces.number,
        Nc = m_mesh->cells.number;

    std::vector<std::vector<int> > faceLil(Nc), cellLil(Nc);

    for(int i=0; i<Nf; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first,
            C1 = m_mesh->faces.connectivity[i].second;

        faceLil[C0].push_back(i);
        cellLil[C0].push_back(C1);

        if(C1!=-1) {
        faceLil[C1].push_back(i);
        cellLil[C1].push_back(C0);
        }
    }

    matrix::makeCRSfromLIL(faceLil, m_adjacency.outerIndex, m_adjacency.faces);
    matrix::makeCRSfromLIL(cellLil, m_adjacency.outerIndex, m_adjacency.cells);

    return 0;
}


int PressureBasedSegregatedSolver::m_buildFaceIterp()
{
    int Nf = m_mesh->faces.number,
        Nc = m_mesh->cells.number,
        Nb = m_boundaries.number;

    using matrix::Triplet;
    std::vector<Triplet<float_t> > triplets;
    triplets.reserve(2*Nf);

    for(int i=Nb; i<Nf; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first,
            C1 = m_mesh->faces.connectivity[i].second;
        float_t wf = m_mesh->faces.wf(i);

        triplets.push_back(Triplet<float_t>(i,C0,wf));
        triplets.push_back(Triplet<float_t>(i,C1,1.0f-wf));
    }

    m_faceInterpMat.resize(Nf,Nc);
    m_faceInterpMat.setFromTriplets(triplets.begin(),triplets.end());

    return 0;
}


int PressureBasedSegregatedSolver::m_buildGradCalc()
{
    int Nf = m_mesh->faces.number,
        Nc = m_mesh->cells.number,
        Nnz = m_adjacency.faces.size();

    m_gradCalcData.ax.resize(Nnz);
    m_gradCalcData.ay.resize(Nnz);
    m_gradCalcData.az.resize(Nnz);
    m_gradCalcData.dir.resize(Nnz);

    for(int i=0; i<Nc; ++i)
    {
        int begin = m_adjacency.outerIndex[i],
            end = m_adjacency.outerIndex[i+1];

        for(int j=begin; j<end; ++j)
        {
            int faceIdx = m_adjacency.faces[j],
                C0 = m_mesh->faces.connectivity[faceIdx].first;

            // determine if signs need to be flipped
            float_t dir = 1-2*(i!=C0);

            m_gradCalcData.ax(j) = dir*m_mesh->faces.area(faceIdx,0);
            m_gradCalcData.ay(j) = dir*m_mesh->faces.area(faceIdx,1);
            m_gradCalcData.az(j) = dir*m_mesh->faces.area(faceIdx,2);
            m_gradCalcData.dir(j) = dir;
        }
    }

    int_t* outerIdxPtr = &m_adjacency.outerIndex[0];
    int_t* innerIdxPtr = &m_adjacency.faces[0];

    m_gradCalcX = mappedCSRf_t(Nc,Nf,Nnz,outerIdxPtr,innerIdxPtr,m_gradCalcData.ax.data());
    m_gradCalcY = mappedCSRf_t(Nc,Nf,Nnz,outerIdxPtr,innerIdxPtr,m_gradCalcData.ay.data());
    m_gradCalcZ = mappedCSRf_t(Nc,Nf,Nnz,outerIdxPtr,innerIdxPtr,m_gradCalcData.az.data());
    m_sumFluxes = mappedCSRf_t(Nc,Nf,Nnz,outerIdxPtr,innerIdxPtr,m_gradCalcData.dir.data());

    return 0;
}


int PressureBasedSegregatedSolver::m_buildMatAssembler()
{
    // here both the structure of the matrix and the auxilary matrices used to
    // set the values of the matrix nz's are prepared

    int Nc = m_mesh->cells.number,
        Nnz = m_adjacency.faces.size();

    // a map relating the final position of the coefficient with the order of
    // its computation is needed
    using matrix::Triplet;
    std::vector<Triplet<int> > triplets;
    triplets.reserve(Nnz+Nc);

    for(int i=0, cursor=0; i<Nc; ++i)
    {
        // diagonal coefficient
        triplets.push_back(Triplet<int>(i,i,cursor++));

        // neighbour coefficients
        for(int j=m_adjacency.outerIndex[i]; j<m_adjacency.outerIndex[i+1]; ++j)
        {
            int cellIdx = m_adjacency.cells[j];
            if(cellIdx!=-1) triplets.push_back(Triplet<int>(i,cellIdx,cursor++));
        }
    }

    // position of nz in matrix = orderMap(computation order)
    std::vector<int> orderMap(triplets.size());
    {
        // temporary matrix with the same structure of the momentum matrix
        matrixCSRi_t tmp(Nc,Nc);
        tmp.setFromTriplets(triplets.begin(),triplets.end());
        const int_t* valPtr = tmp.valuePtr();
        for(int i=0; i<tmp.nonZeros(); ++i) orderMap[valPtr[i]]=i;
    }

    // we keep track of where the diagonal is (for the sum of neighbours)
    // and how to set the anb's
    m_matAssembler.diagonal.resize(Nc);
    m_matAssembler.nz0pos.reserve(Nnz/2);
    m_matAssembler.nz0face.reserve(Nnz/2);
    m_matAssembler.nz1pos.reserve(Nnz/2);
    m_matAssembler.nz1face.reserve(Nnz/2);

    for(int i=0, cursor=0; i<Nc; ++i)
    {
        m_matAssembler.diagonal[i] = orderMap[cursor++];

        for(int j=m_adjacency.outerIndex[i]; j<m_adjacency.outerIndex[i+1]; ++j)
        {
            int faceIdx = m_adjacency.faces[j],
                cellIdx = m_adjacency.cells[j];
            if(cellIdx!=-1)
            {
                int nzpos = orderMap[cursor++];

                // if normal points outwards normal treatment (anb0)
                if(m_gradCalcData.dir(j) > 0.0f) {
                    m_matAssembler.nz0pos.push_back(nzpos);
                    m_matAssembler.nz0face.push_back(faceIdx);
                // else the convective flux needs to be flipped (anb1)
                } else {
                    m_matAssembler.nz1pos.push_back(nzpos);
                    m_matAssembler.nz1face.push_back(faceIdx);
                }
            }
        }
    }
    m_matAssembler.Nnz0 = m_matAssembler.nz0pos.size();
    m_matAssembler.Nnz1 = m_matAssembler.nz1pos.size();
    std::vector<int>().swap(orderMap);

    // set the structure of the system matrices
    m_systemMat.resize(Nc,Nc);
    m_systemMat.setFromTriplets(triplets.begin(),triplets.end());
    std::vector<Triplet<int> >().swap(triplets);
    m_matAssembler.nzPtr = m_systemMat.valuePtr();
    m_matAssembler.momDiagCoeff.resize(Nc);

    return 0;
}


int PressureBasedSegregatedSolver::m_defaultInit()
{
    float_t p0 = -1.0e9f;

    // maximum pressure
    for(int j=0; j<m_boundaries.number; j++)
      if(m_boundaries.type(j)==3)
        p0 = std::max(p0,m_boundaries.conditions(j,3));

    // Set velocity and face fluxes to zero
    m_flwFld_C.u.setZero();
    m_flwFld_C.v.setZero();
    m_flwFld_C.w.setZero();
    m_mdot.setZero();
    // Set pressure to max value
    m_flwFld_C.p.setConstant(p0);

    m_isInit = true;
    return 0;
}


void PressureBasedSegregatedSolver::m_boundaryValues()
{
    for(int j=0; j<m_boundaries.number; j++)
    {
        int C0 = m_mesh->faces.connectivity[j].first;
        switch(m_boundaries.type(j))
        {
          case 1: //Velocity
            m_faceGivenVals.u(j) = m_boundaries.conditions(j,0);
            m_faceGivenVals.v(j) = m_boundaries.conditions(j,1);
            m_faceGivenVals.w(j) = m_boundaries.conditions(j,2);
            m_faceGivenVals.p(j) = m_flwFld_C.p(C0);
            break;
          case 3: //Static pressure
            m_faceGivenVals.u(j) = m_flwFld_C.u(C0);
            m_faceGivenVals.v(j) = m_flwFld_C.v(C0);
            m_faceGivenVals.w(j) = m_flwFld_C.w(C0);
            m_faceGivenVals.p(j) = m_boundaries.conditions(j,3);
            break;
          case 5: //Wall
            m_faceGivenVals.u(j) = 0.0;
            m_faceGivenVals.v(j) = 0.0;
            m_faceGivenVals.w(j) = 0.0;
            m_faceGivenVals.p(j) = m_flwFld_C.p(C0);//+
//                                   m_nabla.p.row(C0).dot(m_mesh->faces.d0.row(j));
            break;
        }
    }
}


void PressureBasedSegregatedSolver::m_computeLimiters(const float_t* phiC_ptr,
                                                      const float_t* phiF_ptr,
                                                      const matrixX3f_t* nablaPhi_ptr,
                                                      float_t* limit_ptr) const
{
    for(int i=0; i<m_mesh->cells.number; ++i)
    {
        int begin = m_adjacency.outerIndex[i],
            end = m_adjacency.outerIndex[i+1];

        float_t min_val=0.0, max_val=0.0, min_prj=-1e-9, max_prj=1e-9, val=phiC_ptr[i];

        for(int j=begin; j<end; ++j)
        {
            int faceIdx = m_adjacency.faces[j],
                cellIdx = m_adjacency.cells[j],
                C0 = m_mesh->faces.connectivity[faceIdx].first;

            float_t delta;
            if(cellIdx!=-1) delta = phiC_ptr[cellIdx]-val;
            else            delta = phiF_ptr[faceIdx]-val;

            min_val = std::min(min_val,delta);
            max_val = std::max(max_val,delta);

            float_t prj;
            if(i==C0) prj = nablaPhi_ptr->row(i).dot(m_mesh->faces.r0.row(faceIdx));
            else      prj = nablaPhi_ptr->row(i).dot(m_mesh->faces.r1.row(faceIdx));

            min_prj = std::min(min_prj,prj);
            max_prj = std::max(max_prj,prj);
        }

        limit_ptr[i] = std::min<float_t>(1.0,std::min(max_val/max_prj,min_val/min_prj));
    }
}


void PressureBasedSegregatedSolver::m_compute2ndOrderFluxes()
{
    int Nf = m_mesh->faces.number,
        Nb = m_boundaries.number;

    // boundary condition fluxes (with corrections)
    for(int i=0; i<Nb; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first;

        // velocity and wall
        if(m_boundaries.type(i)==1 || m_boundaries.type(i)==5)
        {
            m_2ndOrderFluxes(i,0) = -m_mdot(i)*m_flwFld_F.u(i)+m_diffCoeff(i)*
                (m_flwFld_F.u(i)-m_nabla.u.row(C0).dot(m_mesh->faces.d0.row(i)));
            m_2ndOrderFluxes(i,1) = -m_mdot(i)*m_flwFld_F.v(i)+m_diffCoeff(i)*
                (m_flwFld_F.v(i)-m_nabla.v.row(C0).dot(m_mesh->faces.d0.row(i)));
            m_2ndOrderFluxes(i,2) = -m_mdot(i)*m_flwFld_F.w(i)+m_diffCoeff(i)*
                (m_flwFld_F.w(i)-m_nabla.w.row(C0).dot(m_mesh->faces.d0.row(i)));

            m_matAssembler.nzPtr[m_matAssembler.diagonal[C0]]-=
                m_diffCoeff(i)-std::min<float_t>(m_mdot(i),0.0);
        }
    }

    // convective and diffusive corrections
    #define FLUXES(var) -std::max<float_t>(m_mdot(i),0.0)*m_limit.var(C0)*\
                         m_nabla.var.row(C0).dot(m_mesh->faces.r0.row(i))-\
                         std::min<float_t>(m_mdot(i),0.0)*m_limit.var(C1)*\
                         m_nabla.var.row(C1).dot(m_mesh->faces.r1.row(i))+\
                         m_diffCoeff(i)*(\
                         m_nabla.var.row(C1).dot(m_mesh->faces.d1.row(i))-\
                         m_nabla.var.row(C0).dot(m_mesh->faces.d0.row(i)))\

    for(int i=Nb; i<Nf; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first,
            C1 = m_mesh->faces.connectivity[i].second;

        m_2ndOrderFluxes(i,0) = FLUXES(u);
        m_2ndOrderFluxes(i,1) = FLUXES(v);
        m_2ndOrderFluxes(i,2) = FLUXES(w);
    }
    #undef FLUXES
}


int PressureBasedSegregatedSolver::getSolution(MatrixXf* solution_p) const
{
    int Nv = m_mesh->vertices.number,
        Nf = m_mesh->faces.number,
        Nb = m_boundaries.number;

    VectorXf sumOfWeights = VectorXf::Zero(Nv);
    solution_p->setZero(Nv,4);

    for(int fidx=0; fidx<Nf; ++fidx)
    {
        float_t bndFactor = 1.0;
        if(fidx<Nb) if(m_boundaries.type(fidx) < 6) bndFactor = 1000.0;

        for(int j=m_mesh->faces.verticesStart[fidx]; j<m_mesh->faces.verticesStart[fidx+1]; ++j)
        {
            int vidx = m_mesh->faces.verticesIndex[j];

            float_t w = bndFactor/(m_mesh->faces.centroid.row(fidx)-
                                   m_mesh->vertices.coords.row(vidx)).norm();

            sumOfWeights(vidx) += w;
            (*solution_p)(vidx,0) += w*m_flwFld_F.u(fidx);
            (*solution_p)(vidx,1) += w*m_flwFld_F.v(fidx);
            (*solution_p)(vidx,2) += w*m_flwFld_F.w(fidx);
            (*solution_p)(vidx,3) += w*m_flwFld_F.p(fidx);
        }
    }

    for(int i=0; i<Nv; ++i) solution_p->row(i)/=sumOfWeights(i);

    return 0;
}
}
