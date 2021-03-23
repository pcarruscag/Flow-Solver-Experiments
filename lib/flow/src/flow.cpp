//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

#include "../../mathUtils/src/solvers.h"
#include "../../mathUtils/src/preconditioners.h"
#include "../../mathUtils/src/gradients.h"

#include <fstream>
#include <string>
#ifdef __AVX__
#include <x86intrin.h>
#endif

using namespace mathUtils;

namespace flow
{
PressureBasedCoupledSolver::PressureBasedCoupledSolver()
{
    m_ctrlIsSet = false;
    m_meshIsSet = false;
    m_bndrsSet  = false;
    m_isInit    = false;
    m_finished  = false;
}

void PressureBasedCoupledSolver::setControl(const FlowParamManager& flowParams)
{
    m_control = flowParams.m_controlParams;
    m_ctrlIsSet = true;
}

int PressureBasedCoupledSolver::setMesh(vector<UnstructuredMesh>* partitions)
{
    m_meshIsSet = false;
    int partNum = partitions->size();
    if(partNum<1)
        return 1;
    for(int i=0; i<partNum; i++)
        if((*partitions)[i].status()!=0)
            return 2;
    m_partNum = partNum;
    m_part = partitions;
    m_meshIsSet = true;
    return 0;
}

int PressureBasedCoupledSolver::run()
{
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

    cout << std::left << std::setw(90) << "| Allocating memory for run..." << "|\n" << std::flush;
    if(m_allocate()!=0) return 3;

    if(!m_isInit) m_defaultInit();
    m_finished = false;

    // Block preconditioners and linear solvers
    vector<preconditioners::SBILU0<VectorXf> > precond(m_partNum);
    vector<preconditioners::ILU0<VectorXf> > precond_t(m_partNum);
    vector<const preconditioners::PreconditionerBase<SBCSRmatrix<float>,VectorXf>*> precond_p(m_partNum);
    vector<const preconditioners::PreconditionerBase<SparseMatrix<float,RowMajor>,VectorXf>*> precond_t_p(m_partNum);
    solvers::DistBiCGSTAB<SBCSRmatrix<float>,VectorXf,communication,vectorOfMatrix4f> linSolver;
    solvers::DistBiCGSTAB<SparseMatrix<float,RowMajor>,VectorXf,communication,vectorOfMatrix1f> linSolver_t;

    // Vectors of pointers to data inside the ghost cell structure (used by linear solvers)
    vector<const VectorXi*> subMat_nghbrIdx_p(m_partNum);
    vector<const vectorOfMatrix4f*> subMat_coefMat_p(m_partNum);
    vector<const vectorOfMatrix1f*> subMat_coefMat_t_p(m_partNum);

    // Tolerances for the linear solvers
    float linSolTol_vp, linSolTol_t1, linSolTol_t2, linSolTol_min;
    linSolTol_min = m_control.linSolTol;
    linSolTol_vp = linSolTol_t1 = linSolTol_t2 = 0.01f;

    cout << std::left << std::setw(90) << "| Computing wall distance..." << "|\n";
    m_computeWallDistance();

    m_residuals.init(); m_residuals.print(0);

    // BEGIN PARALLEL
    #pragma omp parallel num_threads(m_partNum)
    {
    bool stop = false, diverged = false;
    int prt = omp_get_thread_num(),
        Nc = (*m_part)[prt].cells.number,
        Nf = (*m_part)[prt].faces.number,
        Nb = m_boundaries[prt].number;

    // instantiate the turbulence model objects and pass them pointers to the required solution data
    using turbulenceModels::ModelBase;
    ModelBase* turbModel = ModelBase::makeModel(m_control.turbulenceType);
    turbModel->setup(m_rho, m_mu, &m_flwFld_C[prt].turb1, &m_flwFld_C[prt].turb2, &m_nabla[prt].turb1,
        &m_nabla[prt].turb2, &m_nabla[prt].u, &m_nabla[prt].v, &m_nabla[prt].w,&m_wallDist[prt], &m_mut[prt]);

    // populate the vectors of pointers to preconditioners and ghost cell data
    precond_p[prt]   = &precond[prt];
    precond_t_p[prt] = &precond_t[prt];
    subMat_nghbrIdx_p[prt]  = &m_ghostCells[prt].nghbrCell;
    subMat_coefMat_p[prt]   = &m_ghostCells[prt].coefMat;
    subMat_coefMat_t_p[prt] = &m_ghostCells[prt].coefMat_t;

    // initialization of solution vector, momentum diagonal, and eddy viscosity
    #pragma omp simd
    for(int i=0; i<Nc; ++i)
    {
        m_solution[prt](4*i  ) = m_flwFld_C[prt].u(i);
        m_solution[prt](4*i+1) = m_flwFld_C[prt].v(i);
        m_solution[prt](4*i+2) = m_flwFld_C[prt].w(i);
        m_solution[prt](4*i+3) = m_flwFld_C[prt].p(i);
        m_coefMat[prt].getBlock(m_matDiagMap[prt](i)).diag(2) = 1.0;
    }
    m_mut[prt].setConstant(m_control.initViscRatio*m_mu);

    #pragma omp barrier
    m_updateGhostCells(DIAGONAL,prt);

    // setup linear solvers (after barrier as some vectors of pointers are copied)
    #pragma omp master
    {
        linSolver.setPreconditioner(precond_p);
        linSolver_t.setPreconditioner(precond_t_p);

        linSolver.setData(  &m_coefMat,  &m_source,  &m_solution,  subMat_nghbrIdx_p,subMat_coefMat_p,  &m_communications,4);
        linSolver_t.setData(&m_coefMat_t,&m_source_t,&m_solution_t,subMat_nghbrIdx_p,subMat_coefMat_t_p,&m_communications);
    }

    // some macros to make gradient and limiter computation less verbose
    using mathUtils::gradients::NumBasedBoundaryIndicator;
    NumBasedBoundaryIndicator bndIndicator(Nb);
    #define COMPUTEGRADIENTS(phi)\
    mathUtils::gradients::greenGauss<MatrixX3f,VectorXf,VectorXf,VectorXf,MatrixX3f,NumBasedBoundaryIndicator>\
    (Nc,Nf,(*m_part)[prt].faces.connectivity,bndIndicator,(*m_part)[prt].faces.area,(*m_part)[prt].faces.wf,\
     (*m_part)[prt].cells.volume, m_flwFld_C[prt].phi, m_flwFld_F[prt].phi, m_nabla[prt].phi)

    #define COMPUTELIMITERS(phi)\
    m_computeLimiters(prt, m_flwFld_C[prt].phi, m_ghostCells[prt].flwFld.phi,\
                           m_flwFld_F[prt].phi, m_nabla[prt].phi, m_limit[prt].phi)

    // Begin solver outer iterations
    for(int iter=1; !diverged && ((iter<=m_control.maxIt && !stop) || iter<=m_control.minIt); iter++)
    {
        // relaxation factors
        float relax, relax_t, relax_mu;
        if(iter < m_control.minIt/2) {
            // start with large under-relaxation and laminar flow
            relax = ((m_control.turbulenceType == ModelBase::LAMINAR)?
                      m_control.relax0 : min(m_control.relax0*2.0f,1.0f));
            relax_t = m_control.relax0_t;
            relax_mu= relax_t*(iter-1)*2/m_control.minIt;
        } else if(iter < m_control.minIt) {
            // ramp relaxation factors
            float blend = 2.0f*float(iter)/float(m_control.minIt)-1.0f;
            relax   = blend*m_control.relax1  +(1.0f-blend)*m_control.relax0;
            relax_t = blend*m_control.relax1_t+(1.0f-blend)*m_control.relax0_t;
            relax_mu= relax_t;
        } else {
            // continue ramping viscosity relaxation until it gets to 1
            float blend = 2.0f*float(iter)/(float(m_control.minIt)+1e-6f)-1.0f;
            relax   = m_control.relax1;
            relax_t = m_control.relax1_t;
            relax_mu= min(1.0f,blend*m_control.relax1_t+(1.0f-blend)*m_control.relax0_t);
        }
        if(m_control.turbulenceType == ModelBase::LAMINAR)
            relax_mu = min(1.0f,float(iter-1)*2/m_control.minIt);

        #pragma omp barrier
        m_updateGhostCells(FLOW_FIELD,prt);

        m_boundaryValues(prt,turbModel);

        COMPUTEGRADIENTS(u);  COMPUTEGRADIENTS(v);  COMPUTEGRADIENTS(w);  COMPUTEGRADIENTS(p);
        COMPUTELIMITERS (u);  COMPUTELIMITERS (v);  COMPUTELIMITERS (w);

        if(m_control.turbulenceType != ModelBase::LAMINAR)
        {
            COMPUTEGRADIENTS(turb1);  COMPUTEGRADIENTS(turb2);
            COMPUTELIMITERS (turb1);  COMPUTELIMITERS (turb2);
        }

        turbModel->updateInternalData();
        turbModel->turbulentViscosity(relax_mu,m_mut[prt]);

        #pragma omp barrier
        m_updateGhostCells(GRADS_LIMS,prt);

        // Compute the face mass flows
        m_computeMassFlow(prt);

        // Initialize the source term vector with body force due to rotation
        #pragma omp simd
        for(int i=0; i<Nc; i++)
        {
            float a = m_rho*(*m_part)[prt].cells.volume(i)*m_rotationalSpeed;
            m_source[prt](4*i  ) = a*( 2.0f*m_flwFld_C[prt].v(i)+m_rotationalSpeed*(*m_part)[prt].cells.centroid(i,0));
            m_source[prt](4*i+1) = a*(-2.0f*m_flwFld_C[prt].u(i)+m_rotationalSpeed*(*m_part)[prt].cells.centroid(i,1));
            m_source[prt](4*i+2) = 0.0f;
            m_source[prt](4*i+3) = 0.0f;
        }

        // Build matrix for momentum
        m_assembleMatrix(prt,Nc,Nf,Nb,m_control.cflNumber);

        #pragma omp barrier
        m_updateGhostCells(DIAGONAL,prt);

        // Build matrix for pressure
        m_assembleMatrixP(prt,Nf,Nb);

        // Solve for Velocity and Pressure
        precond[prt].compute(m_coefMat[prt]);
        linSolver.solve(m_control.linSolMaxIt,linSolTol_vp,false);

        #pragma omp master
        {
            if(linSolver.finalResidual()>linSolTol_vp)
                cout << std::left << "| BiCGSTAB did not converge for the VP equation, residual: "
                << std::setw(31) << linSolver.finalResidual() << "|\n";
            linSolTol_vp = max(min(linSolTol_vp,linSolver.initialResidual()*0.05f),linSolTol_min);
            m_residuals.values.block(0,0,4,1) = linSolver.initialVariableResiduals();
        }
        #pragma omp simd
        for(int i=0; i<Nc; i++) m_flwFld_C[prt].u(i)+=relax*(m_solution[prt](4*i  )-m_flwFld_C[prt].u(i));
        #pragma omp simd
        for(int i=0; i<Nc; i++) m_flwFld_C[prt].v(i)+=relax*(m_solution[prt](4*i+1)-m_flwFld_C[prt].v(i));
        #pragma omp simd
        for(int i=0; i<Nc; i++) m_flwFld_C[prt].w(i)+=relax*(m_solution[prt](4*i+2)-m_flwFld_C[prt].w(i));
        #pragma omp simd
        for(int i=0; i<Nc; i++) m_flwFld_C[prt].p(i)+=relax*(m_solution[prt](4*i+3)-m_flwFld_C[prt].p(i));


        // ### TURBULENCE EQUATIONS ###
        if(m_control.turbulenceType != ModelBase::LAMINAR)
        {
        // Initialize source term and solution vector for first turbulence variable
        m_solution_t[prt] = m_flwFld_C[prt].turb1;
        turbModel->rhsSourceTerm(ModelBase::FIRST,m_source_t[prt]);
        for(int i=0; i<Nb; ++i)
            if(m_boundaries[prt].type(i)==BoundaryType::WALL) {
                int C0 = (*m_part)[prt].faces.connectivity[i].first;
                m_source_t[prt](C0) = m_boundaries[prt].conditions(i,2)+
                m_source_t[prt](C0) * m_boundaries[prt].conditions(i,1);
            }
        m_source_t[prt] = m_source_t[prt].cwiseProduct((*m_part)[prt].cells.volume);

        // Build matrix for first turbulence variable
        m_assembleMatrixT(prt,Nf,Nb,static_cast<turbulenceModels::MenterSST*>(turbModel),ModelBase::FIRST);

        // Solve for first turbulence variable
        precond_t[prt].compute(m_coefMat_t[prt]);
        linSolver_t.solve(m_control.linSolMaxIt,linSolTol_t1,false);

        #pragma omp master
        {
            if(linSolver_t.finalResidual()>linSolTol_t1)
                cout << std::left << "| BiCGSTAB did not converge for the T1 equation, residual: "
                << std::setw(31) << linSolver_t.finalResidual() << "|\n";
            linSolTol_t1 = max(min(linSolTol_t1,linSolver_t.initialResidual()*0.1f),linSolTol_min);
            m_residuals.values(4) = linSolver_t.initialResidual(false);
        }
        float max_t1 = turbModel->maxValueOf(ModelBase::FIRST);
        m_flwFld_C[prt].turb1 -= relax_t*(m_flwFld_C[prt].turb1-
                                 m_solution_t[prt].cwiseMax(1.0e-10f).cwiseMin(10.0f*max_t1));

        // Initialize source term and solution vector for second turbulence variable
        m_solution_t[prt] = m_flwFld_C[prt].turb2;
        turbModel->rhsSourceTerm(ModelBase::SECOND,m_source_t[prt]);
        m_source_t[prt] = m_source_t[prt].cwiseProduct((*m_part)[prt].cells.volume);

        // Build matrix for second turbulence variable
        m_assembleMatrixT(prt,Nf,Nb,static_cast<turbulenceModels::MenterSST*>(turbModel),ModelBase::SECOND);

        // Solve for second turbulence variable
        precond_t[prt].compute(m_coefMat_t[prt]);
        linSolver_t.solve(m_control.linSolMaxIt,linSolTol_t2,false);

        #pragma omp master
        {
            if(linSolver_t.finalResidual()>linSolTol_t2)
                cout << std::left << "| BiCGSTAB did not converge for the T2 equation, residual: "
                << std::setw(31) << linSolver_t.finalResidual() << "|\n";
            linSolTol_t2 = max(min(linSolTol_t2,linSolver_t.initialResidual()*0.1f),linSolTol_min);
            m_residuals.values(5) = linSolver_t.initialResidual(false);

            // use master section to print residuals
            if(iter<=3) m_residuals.updateNorms();
            m_residuals.print(iter);
        }
        float max_t2 = turbModel->maxValueOf(ModelBase::SECOND);
        m_flwFld_C[prt].turb2 -= relax_t*(m_flwFld_C[prt].turb2-
                                 m_solution_t[prt].cwiseMax(1.0e-10f).cwiseMin(10.0f*max_t2));
        } else {
        // ### LAMINAR FLOW ###
        #pragma omp master
        {
            // artificially drop the residuals of the turbulent variables
            m_residuals.values.block(4,0,2,1).setConstant((iter==1)? 0.5f : m_control.tol*0.1f);

            // use master section to print residuals
            if(iter<=3) m_residuals.updateNorms();
            m_residuals.print(iter);
        }
        }
        // Convergence criteria
        #pragma omp barrier
        stop = m_residuals.isConverged(m_control.tol);
        diverged = m_residuals.hasDiverged();
    }
    #undef COMPUTEGRADIENTS
    #undef COMPUTELIMITERS
    delete turbModel;

    // keep final residual vector to compare with adjoint
    linSolver.getWorkVector(prt,m_solution[prt]);

    // save the momentum diagonal coefficients in the turbulent source term
    // as they are needed when preparing the data for the adjoint solver
    for(int i=0; i<Nc; ++i)
      m_source_t[prt](i) = m_coefMat[prt].getBlock(m_matDiagMap[prt](i)).diag(2);
    }
    // END PARALLEL

    m_deallocate();
    if(m_residuals.hasDiverged()) cout << std::left << std::setw(90) << "| The flow solver diverged" << "|\n";

    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "|                                                                                         |\n";
    cout << "|                                     SOLVER FINISHED                                     |\n";
    cout << "|                                                                                         |\n";
    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "\n\n" << std::flush;

    m_finished = !m_residuals.hasDiverged();
    return (m_finished? 0:4);
}

int PressureBasedCoupledSolver::m_allocate()
{
    m_flwFld_C.resize(m_partNum);
    m_flwFld_F.resize(m_partNum);
    m_nabla.resize(m_partNum);
    m_limit.resize(m_partNum);
    m_mut.resize(m_partNum);
    m_wallDist.resize(m_partNum);
    m_mdot.resize(m_partNum);
    m_solution.resize(m_partNum);
    m_source.resize(m_partNum);
    m_solution_t.resize(m_partNum);
    m_source_t.resize(m_partNum);

    #pragma omp parallel num_threads(m_partNum)
    {
        int i = omp_get_thread_num();
        int nCells = (*m_part)[i].cells.number,
            nFaces = (*m_part)[i].faces.number;
        m_flwFld_C[i].u.resize(nCells);
        m_flwFld_C[i].v.resize(nCells);
        m_flwFld_C[i].w.resize(nCells);
        m_flwFld_C[i].p.resize(nCells);
        m_flwFld_C[i].turb1.resize(nCells);
        m_flwFld_C[i].turb2.resize(nCells);
        m_flwFld_F[i].u.resize(nFaces);
        m_flwFld_F[i].v.resize(nFaces);
        m_flwFld_F[i].w.resize(nFaces);
        m_flwFld_F[i].p.resize(nFaces);
        m_flwFld_F[i].turb1.resize(nFaces);
        m_flwFld_F[i].turb2.resize(nFaces);
        m_nabla[i].u.setZero(nCells,3);
        m_nabla[i].v.setZero(nCells,3);
        m_nabla[i].w.setZero(nCells,3);
        m_nabla[i].p.setZero(nCells,3);
        m_nabla[i].turb1.setZero(nCells,3);
        m_nabla[i].turb2.setZero(nCells,3);
        m_limit[i].u.setZero(nCells);
        m_limit[i].v.setZero(nCells);
        m_limit[i].w.setZero(nCells);
        m_limit[i].turb1.setZero(nCells);
        m_limit[i].turb2.setZero(nCells);
        m_mut[i].resize(nCells);
        m_wallDist[i].resize(nCells);
        m_mdot[i].setZero(nFaces);
        m_solution[i].resize(4*nCells);
        m_source[i].resize(4*nCells);
        m_solution_t[i].resize(nCells);
        m_source_t[i].resize(nCells);
    }
    m_matrixIndexes(); // create the coefficient matrices

    return 0;
}

int PressureBasedCoupledSolver::m_deallocate()
{
    // the limits, solution, and matrices are used by the adjoint solver
    // the turbulent source term is used to store the momentum diagonal
//    vector<flowField>().swap(m_limit);
//    vector<SBCSRmatrix<float> >().swap(m_coefMat);
//    vector<SparseMatrix<float,RowMajor> >().swap(m_coefMat_t);
//    vector<VectorXf>().swap(m_solution);
//    vector<VectorXf>().swap(m_source_t);

    vector<VectorXi>().swap(m_matDiagMap);
    vector<MatrixX2i>().swap(m_matOffMap);
    vector<VectorXf>().swap(m_source);

    return 0;
}

float PressureBasedCoupledSolver::m_areaOfGroup(const int part, const int group) const
{
    float area = 0.0;
    for(auto face : (*m_part)[part].faces.groups[group].second)
        area += (*m_part)[part].faces.area.row(face).norm();
    return area;
}

void PressureBasedCoupledSolver::m_bndrsOfType(const int part,
                                               const int type,
                                               vector<int>& faces,
                                               const bool append) const
{
    if(!append)
        faces.clear();
    for(int i=0; i<m_boundaries[part].number; i++)
        if(m_boundaries[part].type(i)==type)
            faces.push_back(i);
}

int PressureBasedCoupledSolver::m_defaultInit()
{
    float turb1 = 0.0, turb2 = 0.0, p0 = -1.0e9f;
    int counter = 0;

    #pragma omp parallel num_threads(m_partNum) reduction(+:turb1,turb2,counter) reduction(max:p0)
    {
        int i = omp_get_thread_num();
        // average turbulence variables
        for(int j=0; j<m_boundaries[i].number; j++)
            if(m_boundaries[i].type(j)==BoundaryType::VELOCITY ||
               m_boundaries[i].type(j)==BoundaryType::TOTALPRESSURE) {
                turb1 += m_boundaries[i].conditions(j,4);
                turb2 += m_boundaries[i].conditions(j,5);
                counter++;
            }
        // maximum pressure
        for(int j=0; j<m_boundaries[i].number; j++)
            if(m_boundaries[i].type(j)==BoundaryType::PRESSURE ||
               m_boundaries[i].type(j)==BoundaryType::MASSFLOW)
                p0 = std::max(p0,m_boundaries[i].conditions(j,3));
    }
    turb1 /= counter;
    turb2 /= counter;
    #pragma omp parallel num_threads(m_partNum)
    {
        int i = omp_get_thread_num();
        // Set velocity to zero
        m_flwFld_C[i].u.setZero();
        m_flwFld_C[i].v.setZero();
        m_flwFld_C[i].w.setZero();
        // Set turbulence to average values
        m_flwFld_C[i].turb1.setConstant(turb1);
        m_flwFld_C[i].turb2.setConstant(turb2);
        // Set pressure to max value
        m_flwFld_C[i].p.setConstant(p0);
    }
    m_isInit = true;
    return 0;
}

void PressureBasedCoupledSolver::m_computeWallDistance()
{
    ArrayXi wallNum(m_partNum);
    MatrixX3f wallCentroid;
    #pragma omp parallel num_threads(m_partNum)
    {
        // fetch coordinates of wall centers into a shared array
        int prt = omp_get_thread_num();
        std::vector<int> walls;
        m_bndrsOfType(prt,BoundaryType::WALL,walls);
        wallNum(prt) = walls.size();
        #pragma omp barrier
        int Nw = wallNum.sum();
        #pragma omp master
        wallCentroid.resize(Nw,NoChange);
        #pragma omp barrier
        int start = wallNum.head(prt).sum();
        for(size_t i=0; i<walls.size(); ++i)
            wallCentroid.row(start+i) = (*m_part)[prt].faces.centroid.row(walls[i]);
        #pragma omp barrier

        // get minimum distance for each cell
        int Nc = (*m_part)[prt].cells.number;
        const float largeNum = 1.0e9f;
        const float* Xdomain = (*m_part)[prt].cells.centroid.data();
        const float* Xwall   = wallCentroid.data();
        float*       Dwall   = m_wallDist[prt].data();

        #ifdef __AVX__
        int Nc_simd = (Nc/8)*8;
        for(int i=0; i<Nc_simd; i+=8)
        {
            // load the coordinates of domain point(s) and initialize
            // the minimum distance (to the wall) to a large number
            __m256
            dmin = _mm256_broadcast_ss(&largeNum),
            x = _mm256_loadu_ps(&Xdomain[  i   ]),
            y = _mm256_loadu_ps(&Xdomain[ i+Nc ]),
            z = _mm256_loadu_ps(&Xdomain[i+Nc*2]);

            // find minimum squared distance from domain to wall
            for(int j=0; j<Nw; ++j)
            {
                __m256 d,
                t = _mm256_sub_ps(x, _mm256_broadcast_ss(&Xwall[  j   ]));
                d = _mm256_mul_ps(t,t);     // d  = (xa-xb)^2
                t = _mm256_sub_ps(y, _mm256_broadcast_ss(&Xwall[ j+Nw ]));
                d = _mm256_fmadd_ps(t,t,d); // d += (ya-yb)^2
                t = _mm256_sub_ps(z, _mm256_broadcast_ss(&Xwall[j+2*Nw]));
                d = _mm256_fmadd_ps(t,t,d); // d += (za-zb)^2

                dmin = _mm256_min_ps(dmin,d);
            }

            // square root and store result
            dmin = _mm256_sqrt_ps(dmin);
            _mm256_storeu_ps(&Dwall[i],dmin);
        }
        #else
        int Nc_simd = 0;
        #endif
        // remainder of loop
        for(int i=Nc_simd; i<Nc; ++i)
        {
            float dmin = largeNum;
            for(int j=0; j<Nw; ++j)
                dmin = min(dmin,powf(Xdomain[  i   ]-Xwall[  j   ],2.0f)+
                                powf(Xdomain[ i+Nc ]-Xwall[ j+Nw ],2.0f)+
                                powf(Xdomain[i+2*Nc]-Xwall[j+2*Nw],2.0f));
            Dwall[i] = sqrt(dmin);
        }

        // wall distance for wall cells measured in the normal direction
        for(auto k : walls) {
            int C0 = (*m_part)[prt].faces.connectivity[k].first;
            m_wallDist[prt](C0) = min(m_wallDist[prt](C0),
                (*m_part)[prt].faces.alfa0(k)*(*m_part)[prt].faces.area.row(k).norm());
        }
    }
}

int PressureBasedCoupledSolver::m_computeMassFlow(const int prt)
{
    auto& faces = (*m_part)[prt].faces;
    auto& cells = (*m_part)[prt].cells;

    // Linear interpolation of cell velocities
    m_mdot[prt] = faces.area.col(0).cwiseProduct(m_flwFld_F[prt].u)+
                  faces.area.col(1).cwiseProduct(m_flwFld_F[prt].v)+
                  faces.area.col(2).cwiseProduct(m_flwFld_F[prt].w);

    // Added dissipation
    int Nb = m_boundaries[prt].number,
        Nf = faces.number;

    Vector3f nabla_pf, r1_prime;
    float Vf, cf, af;

    for(int i=Nb; i<Nf; ++i)
    {
        int C0 = faces.connectivity[i].first,
            C1 = faces.connectivity[i].second;

        auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));
        auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));

        nabla_pf = faces.wf(i) *m_nabla[prt].p.row(C0)+
             (1.0f-faces.wf(i))*m_nabla[prt].p.row(C1);

        Vf = cells.volume(C0)+cells.volume(C1);
        cf = B00.diag(2)+B11.diag(2);
        af = Vf/(cf*(faces.alfa0(i)-faces.alfa1(i)));

        m_mdot[prt](i) += af*(m_flwFld_C[prt].p(C0)+nabla_pf.dot(faces.r0.row(i))-
                              m_flwFld_C[prt].p(C1)-nabla_pf.dot(faces.r1.row(i)));
    }

    for(int i=0; i<Nb; ++i)
    {
        int C0 = faces.connectivity[i].first, C1,
            bndType = m_boundaries[prt].type(i);

        auto B00 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C0));

        if(bndType == BoundaryType::PRESSURE || bndType == BoundaryType::MASSFLOW)
        {
            Vf = cells.volume(C0);
            cf = B00.diag(2);
            af = Vf/(cf*faces.alfa0(i));

            m_mdot[prt](i) -= af*(m_flwFld_F[prt].p(i)-m_flwFld_C[prt].p(C0)-
                                  m_nabla[prt].p.row(C0).dot(faces.r0.row(i)));
        }
        else if(bndType == BoundaryType::PERIODIC && m_boundaries[prt].conditions(i,1) == 1)
        {
            C1 = m_boundaries[prt].conditions(i,0); // Index of matching cell

            auto B11 = m_coefMat[prt].getBlock(m_matDiagMap[prt](C1));

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

            Vf = cells.volume(C0)+cells.volume(C1);
            cf = B00.diag(2)+B11.diag(2);
            af = Vf/(cf*(faces.alfa0(i)-faces.alfa1(i)));

            m_mdot[prt](i) += af*(m_flwFld_C[prt].p(C0)+nabla_pf.dot(faces.r0.row(i))-
                                  m_flwFld_C[prt].p(C1)-nabla_pf.dot(r1_prime));
        }
        else if(bndType == BoundaryType::GHOST)
        {
            C1 = m_boundaries[prt].conditions(i,0); // Index of ghost cell

            nabla_pf = faces.wf(i) *m_nabla[prt].p.row(C0)+
                 (1.0f-faces.wf(i))*m_ghostCells[prt].nabla.p.row(C1);

            Vf = cells.volume(C0)+m_ghostCells[prt].vol_nb(C1);
            cf = B00.diag(2)+m_ghostCells[prt].diag_nb(C1);
            af = Vf/(cf*(faces.alfa0(i)-faces.alfa1(i)));

            m_mdot[prt](i) += af*(m_flwFld_C[prt].p(C0)+nabla_pf.dot(faces.r0.row(i))-
                         m_ghostCells[prt].flwFld.p(C1)-nabla_pf.dot(faces.r1.row(i)));
        }
    }

    m_mdot[prt] *= m_rho;

    return 0;
}

void PressureBasedCoupledSolver::residuals::print(const int iteration)
{
    if(((iteration-1)%10==0 || iteration==0) && iteration!=1) // print the header every 10 iterations
    {
        cout << "-------------------------------------------------------------------------------------------\n";
        cout << "| Iteration |   U Mom.   |   V Mom.   |   W Mom.   |  Pressure  |   Turb. 1  |   Turb. 2  |\n";
        cout << "-------------------------------------------------------------------------------------------\n";
    }
    if(iteration > 0)
    {
        cout << std::right << "|" << std::setw(10) << iteration << " |";
        for(int i=0; i<6; ++i)
            cout << std::setw(11) << values(i)/norms(i) << " |";
        cout << '\n';
    }
    cout << std::flush;
}

}
