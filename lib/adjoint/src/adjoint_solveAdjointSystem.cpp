//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjoint.h"

#include "../../mathUtils/src/preconditioners.h"
#include "../../mathUtils/src/solvers.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

using std::vector;
using std::cout;
using std::endl;

using namespace mathUtils;

namespace adjoint
{
template<typename RHS_t>
int PressureBasedCoupledSolverAdjoint::m_solveAdjointSystem()
{
    using Scalar_t = typename RHS_t::Scalar;
    bool rightPrec = false;
    m_control.getRightPrec(rightPrec);

    m_tSolve -= omp_get_wtime();
    m_flowJacobian.timer = 0.0;
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    cout << "|                                ADJOINT  SYSTEM  SOLUTION                                |" << endl;
    cout << "|-----------------------------------------------------------------------------------------|" << endl;
    cout << "| Iteration | Res. (MAX) | Res. (AVG) | Adj. UVW (AVG) | Adj. P (AVG) | Linear Solver I/R |" << endl;
    cout << "|-----------------------------------------------------------------------------------------|" << endl;

    int maxIters, linSolMaxIt;
    Scalar_t alphaMin, alphaMax, tolerance, linSolTol, linSolMinTol, linSolRatTol;
    m_control.getStopCriteria(maxIters,tolerance);
    m_control.getRelaxLimits(alphaMin,alphaMax);
    m_control.getLinSolParam(linSolMaxIt,linSolTol,linSolMinTol,linSolRatTol);
    // following used to measure the change of the adjoint variables
    long double avgAdjV, avgAdjP; numPrec avgMult = numPrec(1)/(m_inData_p->cellNumber*m_objNum);

    // setup the solver for A^T
    vector<const VectorXi*> Aij_ptr(m_partNum);
    vector<const vector<Matrix<Scalar_t,4,4>,aligned_allocator<Matrix<Scalar_t,4,4> > >*> Aij(m_partNum);
    vector<preconditioners::SBILU0<RHS_t> > precond(m_partNum);
    vector<const preconditioners::PreconditionerBase<SBCSRmatrix<Scalar_t>,RHS_t>*> precond_p(m_partNum);
    vector<RHS_t> rhs(m_partNum), lhs(m_partNum);
    ArrayXi partSize(m_partNum);
    #pragma omp parallel num_threads(m_partNum)
    {
        int prt = omp_get_thread_num();

        precond[prt].compute(m_inData_p->Aii[prt]);
        precond_p[prt] = &precond[prt];

        Aij_ptr[prt] = &m_inData_p->Aij_ptr[prt];
        Aij[prt] = &m_inData_p->Aij[prt];

        partSize(prt) = 4*m_inData_p->Aii[prt].rows();
        rhs[prt].resize(partSize(prt),m_objNum);
        lhs[prt].resize(partSize(prt),m_objNum);
    }
    solvers::DistBiCGSTAB<SBCSRmatrix<Scalar_t>,RHS_t,PressureBasedCoupledSolver::communication,
                      vector<Matrix<Scalar_t,4,4>,aligned_allocator<Matrix<Scalar_t,4,4> > > > solver;
    solver.setData(&m_inData_p->Aii,&rhs,&lhs,Aij_ptr,Aij,&m_inData_p->communications,4);
    solver.setPreconditioner(precond_p);

    // solve for the multiple rhs's
    Array<Scalar_t,1,Dynamic> resNorm(m_objNum), residual(m_objNum);
    resNorm = m_objFlowJacobian.cast<Scalar_t>().colwise().norm();
    resNorm = resNorm.max(Array<Scalar_t,1,Dynamic>::Constant(1,m_objNum,1e-9));

    // 1 - initialize adjoint vars and solver rhs with the adjoint residual
    #define DECLLOCALVARS int prt = omp_get_thread_num(),\
                              start = partSize.head(prt).sum(),\
                              numEl = partSize(prt)
    if(!rightPrec)
    {
        m_flowJacobian.product = -m_objFlowJacobian;
        if(m_isInit) m_flowJacobian.gemm(m_objNum,m_adjointVars.data());
        else         m_adjointVars.setZero(4*m_inData_p->cellNumber,m_objNum);

        #pragma omp parallel num_threads(m_partNum)
        {
        DECLLOCALVARS;
        rhs[prt] = -m_flowJacobian.product.cast<Scalar_t>().block(start,0,numEl,m_objNum);
        }
    }
    else {
        m_adjointVars = m_objFlowJacobian;
        #pragma omp parallel num_threads(m_partNum)
        lhs[omp_get_thread_num()].setZero();
    }

    for(int iter=0; iter<maxIters; ++iter)
    {
        #pragma omp parallel num_threads(m_partNum)
        {
        DECLLOCALVARS;
        if(rightPrec) {
            // 2 - Apply preconditioner on the right
            rhs[prt] = m_adjointVars.cast<Scalar_t>().block(start,0,numEl,m_objNum);
        }
        else {
            // 2 - compute delta_lambda i.e. lhs
            lhs[prt].setZero();
        }
        solver.solve(linSolMaxIt,linSolTol,false);

        // 3 - swap lambda and delta_lambda to calculate the
        //     product of the latter with the jacobian
        for(int j=0; j<m_objNum; ++j)
          for(int i=0; i<numEl; ++i) {
            numPrec tmp = m_adjointVars(start+i,j);
            m_adjointVars(start+i,j) = lhs[prt](i,j);
            lhs[prt](i,j) = tmp;
          }
        }

        Matrix<Scalar_t,Dynamic,Dynamic> reduxVar; reduxVar.setZero(m_partNum,m_objNum);

        if(!rightPrec)
        {
            // 4 - compute d, the product of delta_lambda with the jacobian
            //     (stored in product member var that needs to be cleared explicitly)
            m_flowJacobian.product.setZero();
            m_flowJacobian.gemm(m_objNum,m_adjointVars.data());

            // 5 - compute alpha, initialized with the denominator (d,d)
            Matrix<Scalar_t,1,Dynamic> alpha = m_flowJacobian.product.cast<Scalar_t>().colwise().squaredNorm();
            // numerator (r,d) in reduxVar
            avgAdjV = avgAdjP = 0.0;
            #pragma omp parallel num_threads(m_partNum) reduction(+:avgAdjV,avgAdjP)
            {
            DECLLOCALVARS;
            for(int j=0; j<m_objNum; ++j)
              for(int i=0; i<numEl; ++i)
                reduxVar(prt,j) += rhs[prt](i,j)*static_cast<Scalar_t>(m_flowJacobian.product(start+i,j));

            #pragma omp barrier
            #pragma omp master
            {
            alpha = reduxVar.colwise().sum().cwiseQuotient(alpha);
            for(int j=0; j<m_objNum; ++j)
              alpha(j) = std::min(std::max(alpha(j),alphaMin),alphaMax);
            }
            #pragma omp barrier

            // 6 - update lambda, this reverts the swap done in 3)
            for(int j=0; j<m_objNum; ++j)
              for(int i=0; i<numEl; ++i) {
                m_adjointVars(start+i,j) = lhs[prt](i,j)+alpha(j)*static_cast<Scalar_t>(m_adjointVars(start+i,j));
                if(i%4==3) avgAdjP += std::abs(avgMult*m_adjointVars(start+i,j));
                else       avgAdjV += std::abs(avgMult*m_adjointVars(start+i,j))/3;
              }

            // 7 - update rhs (the adjoint residual)
            for(int j=0; j<m_objNum; ++j)
              for(int i=0; i<numEl; ++i)
                rhs[prt](i,j) -= alpha(j)*static_cast<Scalar_t>(m_flowJacobian.product(start+i,j));

            // 8 - convergence criteria
            reduxVar.row(prt) = rhs[prt].colwise().squaredNorm();
            }
        }
        else
        {
            // 4 - compute the adjoint residual
            m_flowJacobian.product = -m_objFlowJacobian;
            m_flowJacobian.gemm(m_objNum,m_adjointVars.data());

            avgAdjV = avgAdjP = 0.0;
            #pragma omp parallel num_threads(m_partNum) reduction(+:avgAdjV,avgAdjP)
            {
            DECLLOCALVARS;
            // 5 - update lambda, this reverts the swap done in 3)
            for(int j=0; j<m_objNum; ++j)
              for(int i=0; i<numEl; ++i) {
                numPrec tmp = m_adjointVars(start+i,j);
                m_adjointVars(start+i,j) = lhs[prt](i,j)-alphaMin*static_cast<Scalar_t>(m_flowJacobian.product(start+i,j));
                lhs[prt](i,j) = tmp;
                if(i%4==3) avgAdjP += std::abs(avgMult*m_adjointVars(start+i,j));
                else       avgAdjV += std::abs(avgMult*m_adjointVars(start+i,j))/3;
              }

            // 6 - update rhs (the adjoint residual)
            for(int j=0; j<m_objNum; ++j)
              for(int i=0; i<numEl; ++i)
                rhs[prt](i,j) = static_cast<Scalar_t>(m_flowJacobian.product(start+i,j));

            // 7 - convergence criteria
            reduxVar.row(prt) = rhs[prt].colwise().squaredNorm();
            }
        }

        residual = reduxVar.colwise().sum().array().sqrt()/resNorm;
        // renormalize if residual increases on first iteration
        for(int j=0; j<m_objNum && iter==0; ++j)
          if(residual(j)>Scalar_t(10)) {
            resNorm(j)  = std::sqrt(reduxVar.col(j).sum());
            residual(j) = 1.0;
          }

        cout << std::right
             <<  "|" << std::setw(10) << iter
             << " |" << std::setw(11) << residual.maxCoeff()
             << " |" << std::setw(11) << residual.mean()
             << " |" << std::setw(15) << avgAdjV
             << " |" << std::setw(13) << avgAdjP
             << " |" << std::setw( 5) << solver.numIters() << std::left
             << " / "<< std::setw(11) << solver.finalResidual()
             <<  "|" << endl;

        if((residual<tolerance).all()) break;

        // set new tolerance for linear solver
        linSolTol = std::min(linSolTol,linSolRatTol*residual.maxCoeff());
        linSolTol = std::max(linSolTol,linSolMinTol);
    }

    if(rightPrec)
    {
        #pragma omp parallel num_threads(m_partNum)
        {
        DECLLOCALVARS;
        rhs[prt] = m_adjointVars.cast<Scalar_t>().block(start,0,numEl,m_objNum);
        solver.solve(linSolMaxIt,linSolTol,false);

        for(int j=0; j<m_objNum; ++j)
          for(int i=0; i<numEl; ++i)
            m_adjointVars(start+i,j) = lhs[prt](i,j);
        }
    }

    cout << "|-----------------------------------------------------------------------------------------|" << endl;
    #undef DECLLOCALVARS
    m_tSolve += omp_get_wtime()-m_flowJacobian.timer;
    m_tFlwJac += m_flowJacobian.timer;
    return 0;
}
using prec = AdjointInputData::solvePrec;
#define SOLVEADJOINTSYSTEMINST(T,N,S) \
template int PressureBasedCoupledSolverAdjoint::m_solveAdjointSystem<Matrix<T,Dynamic,N,S> >()
SOLVEADJOINTSYSTEMINST(prec,1,ColMajor);
SOLVEADJOINTSYSTEMINST(prec,2,ColMajor);
SOLVEADJOINTSYSTEMINST(prec,3,ColMajor);
SOLVEADJOINTSYSTEMINST(prec,4,ColMajor);
SOLVEADJOINTSYSTEMINST(prec,Dynamic,ColMajor);
#undef SOLVEADJOINTSYSTEMINST
}
