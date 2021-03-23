//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjoint.h"

#include "../../mathUtils/src/matrix.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using std::vector;
using std::pair;
using std::cout;
using std::endl;

using namespace mathUtils;

namespace adjoint
{
PressureBasedCoupledSolverAdjoint::PressureBasedCoupledSolverAdjoint()
{
    m_dataIsSet = false;
    m_isInit    = false;
    m_finished  = false;
    m_firstOrder = false;
}

int PressureBasedCoupledSolverAdjoint::setData(const flow::AdjointInputData* inData_p,
                                               fileManagement::AdjointParamManager control)
{
    if(!inData_p || !control.dataIsReady()) return 1;

    m_control   = control;
    m_inData_p  = inData_p;
    m_dataIsSet = true;
    m_isInit    = false;
    m_finished  = false;
    m_partNum   = omp_get_max_threads();
    m_control.getObjectives(m_objectiveList);
    m_objNum = m_objectiveList.size();

    return 0;
}

int PressureBasedCoupledSolverAdjoint::initialize(const std::string &filePath)
{
    if(!m_dataIsSet) return 1;

    std::ifstream file;
    file.open(filePath.c_str(),std::ios::binary);
    if(!file.good()) return 2;

    m_isInit = false;

    int nRows = 4*m_inData_p->cellNumber, row=0;
    m_adjointVars.resize(nRows,m_objNum);

    for(int col=0; col<m_objNum; ++col)
      for(row=0; row<nRows && !file.eof(); ++row) {
        char data[4];
        file.read(data,4);
        m_adjointVars(row,col) = *reinterpret_cast<float*>(data);
      }
    file.close();
    if(row!=nRows) return 3;
    m_isInit = true;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::run()
{
    if(!m_dataIsSet)
        return 1;
    m_finished = false;
    m_control.getFirstOrder(m_firstOrder);
    m_tFlwJac = m_tGeoJac = m_tObjJac = m_tSolve = 0.0;

    cout << std::scientific;
    cout.precision(3);
    cout << endl << endl;
    cout << "-------------------------------------------------------------------------------------------" << endl;
    cout << "|                                                                                         |" << endl;
    cout << "|                                ADJOINT  SOLVER  STARTED                                 |" << endl;
    cout << "|                                                                                         |" << endl;
    cout << "-------------------------------------------------------------------------------------------" << endl;

    m_buildConnectivity();

    m_computeObjective(); // jacobians, flow and geometry

    m_allocateFlowJacobian();

    m_computeResidualJacobianFlw();

    // free the memory used by cell based connectivity, not needed during solution
    vector<vector<int> >().swap(m_connectivity_F);

    #define solve(N,S) m_solveAdjointSystem< Matrix<AdjointInputData::solvePrec,Dynamic,N,S> >()
    switch(m_objNum) {
    case 1:  solve(1,ColMajor); break;
    case 2:  solve(2,ColMajor); break;
    case 3:  solve(3,ColMajor); break;
    case 4:  solve(4,ColMajor); break;
    default: solve(Dynamic,ColMajor); // generic case
    }
    #undef solve

    m_flowJacobian.free();

    m_buildConnectivity();

    m_computeResidualJacobianGeo();

    cout << std::fixed << std::left;
    cout.precision(2);
    cout << std::setw(90) << "| Timer Information [s]" << "|" << endl;
    cout << "|  Objective function : " << std::setw(66) << m_tObjJac << "|" << endl;
    cout << "|  Flow jacobian ops. : " << std::setw(66) << m_tFlwJac << "|" << endl;
    cout << "|  Linear solver ops. : " << std::setw(66) << m_tSolve  << "|" << endl;
    cout << "|  Geometry jacobian  : " << std::setw(66) << m_tGeoJac << "|" << endl;
    cout << std::setw(90) << "|" << "|" << endl;
    cout << "-------------------------------------------------------------------------------------------" << endl;
    cout << "|                                                                                         |" << endl;
    cout << "|                                ADJOINT  SOLVER  FINISHED                                |" << endl;
    cout << "|                                                                                         |" << endl;
    cout << "-------------------------------------------------------------------------------------------" << endl;
    cout << endl << endl;

    m_finished = true;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::getObjectiveVals(Matrix<numPrec,Dynamic,1>* values) const
{
    if(!m_finished) return 1;
    *values = m_objValues;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::getDerivatives(Matrix<numPrec,Dynamic,Dynamic>* gradient) const
{
    if(!m_finished) return 1;
    *gradient = m_objGradient;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::getAdjointVars(Matrix<numPrec,Dynamic,Dynamic>* adjVars) const
{
    if(!m_finished) return 1;
    *adjVars = m_adjointVars;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::getObjectiveJacobian(const VariableType varType,
                                                Matrix<numPrec,Dynamic,Dynamic>* jacobian) const
{
    if(!m_finished) return 1;
    if(varType==GEOMETRY)  *jacobian = m_objGeoJacobian;
    else if(varType==FLOW) *jacobian = m_objFlowJacobian;
    else return 2;
    return 0;
}

int PressureBasedCoupledSolverAdjoint::saveState(const std::string &filePath) const
{
    if(!m_finished) return 1;

    std::ofstream file;
    file.open(filePath.c_str(),std::ios::binary);
    if(!file.is_open()) return 2;

    #define WRITE(V) {float _v = V; file.write(reinterpret_cast<char *>(&_v),4);}
    for(int col=0; col<m_objNum; ++col)
      for(int row=0; row<m_adjointVars.rows(); ++row)
        WRITE(m_adjointVars(row,col))
    file.close();
    #undef WRITE
    return 0;
}

void PressureBasedCoupledSolverAdjoint::m_buildConnectivity()
{
    int Nc = m_inData_p->cellNumber,
        Nf = m_inData_p->faceNumber,
        Nb = m_inData_p->boundaryNumber,
        C0, C1;
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    cout << std::left << std::setw(90) << "| Building cell based connectivity..." << "|" << endl;

    m_connectivity_F.resize(Nc);
    // internal faces
    for(int i=Nb; i<Nf; ++i)
    {
        C0 = m_inData_p->connectivity[i].first;
        C1 = m_inData_p->connectivity[i].second;
        m_connectivity_F[C0].push_back(i);
        m_connectivity_F[C1].push_back(i);
    }
    // boundaries
    for(int i=0; i<Nb; ++i)
    {
        C0 = m_inData_p->connectivity[i].first;
        m_connectivity_F[C0].push_back(i);
    }
}

int PressureBasedCoupledSolverAdjoint::m_computeResidualJacobianFlw()
{
    m_tFlwJac -= omp_get_wtime();
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    cout << "|                           FLOW  JACOBIAN  MATRIX  CALCULATION                           |" << endl;

    int Nc = m_inData_p->cellNumber;
    VectorXf residual(4*Nc), residualNorm(4*Nc);

    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,1000)
    for(int i=0; i<Nc; ++i)
    {
        Vector4f residual_i, residualNorm_i;
        vector<int> cells, vertices, cellDegEndIdx;
        Matrix<numPrec,Dynamic,4> drdw;

        m_computeResidualForCell<FLOW>(i,residual_i,residualNorm_i,cells,cellDegEndIdx,vertices,drdw);

        residual.block(4*i,0,4,1) = -1.0f*residual_i;
        residualNorm.block(4*i,0,4,1) = residualNorm_i;

        int outerIdxStart = m_flowJacobian.outerIdxPtr(i),
            outerIdxSize  = m_flowJacobian.outerIdxPtr(i+1)-outerIdxStart;

        #pragma omp simd
        for(int j=0; j<cellDegEndIdx[2]; ++j)
            m_flowJacobian.cellIdx(outerIdxStart+j) = cells[j];

        m_flowJacobian.coefficients.block(4*outerIdxStart,0,4*outerIdxSize,4) = drdw;
    }

    // Statistics on residual calculation
    {
    int row, col;
    float rms_res, max_res;

    rms_res = m_inData_p->residuals.cwiseQuotient(residualNorm).norm()*float(0.5/std::sqrt(Nc));
    max_res = m_inData_p->residuals.cwiseQuotient(residualNorm).cwiseAbs().maxCoeff(&row,&col);

    cout << std::left << std::setw(90) << "| Residuals from Flow Solver" << "|" << endl;
    cout << "|  RMS: " << std::left << std::setw(10) << rms_res;
    cout << "   MAX: " << std::left << std::setw(10) << max_res;
    cout << "   @ cell: " << std::left << std::setw(10) << row/4;
    cout << "   @ eq: " << std::left << std::setw(10) << row%4 << std::right << std::setw(15) << "|" << endl;

    rms_res = residual.cwiseQuotient(residualNorm).norm()*float(0.5/std::sqrt(Nc));
    max_res = residual.cwiseQuotient(residualNorm).cwiseAbs().maxCoeff(&row,&col);

    cout << std::left << std::setw(90) << "| Residuals from Adjoint Solver" << "|" << endl;
    cout << "|  RMS: " << std::left << std::setw(10) << rms_res;
    cout << "   MAX: " << std::left << std::setw(10) << max_res;
    cout << "   @ cell: " << std::left << std::setw(10) << row/4;
    cout << "   @ eq: " << std::left << std::setw(10) << row%4 << std::right << std::setw(15) << "|" << endl;

    rms_res = (residual-m_inData_p->residuals).cwiseQuotient(residualNorm).norm()*float(0.5/std::sqrt(Nc));
    max_res = (residual-m_inData_p->residuals).cwiseQuotient(residualNorm).cwiseAbs().maxCoeff(&row,&col);

    cout << std::left << std::setw(90) << "| Difference" << "|" << endl;
    cout << "|  RMS: " << std::left << std::setw(10) << rms_res;
    cout << "   MAX: " << std::left << std::setw(10) << max_res;
    cout << "   @ cell: " << std::left << std::setw(10) << row/4;
    cout << "   @ eq: " << std::left << std::setw(10) << row%4 << std::right << std::setw(15) << "|" << endl;
    }
    m_tFlwJac += omp_get_wtime();
    return 0;
}

int PressureBasedCoupledSolverAdjoint::m_computeResidualJacobianGeo()
{
    m_tGeoJac -= omp_get_wtime();
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    cout << "|                         GEOMETRY  JACOBIAN  MATRIX  CALCULATION                         |" << endl;
    cout << "|  0% |                                 Progress                                  | 100%  |" << endl;
    cout << "|     |" << std::flush;

    int Nc = m_inData_p->cellNumber,
        Nv = m_inData_p->vertexNumber;

    m_objGradient = m_objGeoJacobian;

    vector<omp_lock_t> locks(Nv);

    for(int i=0; i<Nv; ++i) omp_init_lock(&locks[i]);

    float progressFactor = 75.0/Nc;
    int progressMonitor = 1, nTicksPrinted = 0;
    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,1000)
    for(int i=0; i<Nc; ++i)
    {
        int prt = omp_get_thread_num();

        Vector4f residual_i, residualNorm_i;
        vector<int> cells, vertices, cellDegEndIdx;
        Matrix<numPrec,Dynamic,4> drdz;

        m_computeResidualForCell<GEOMETRY>(i,residual_i,residualNorm_i,cells,cellDegEndIdx,vertices,drdz);

        Matrix<numPrec,Dynamic,Dynamic> tmp = drdz*m_adjointVars.block(4*i,0,4,m_objNum);

        for(int j=0; j<int(vertices.size()); ++j)
        {
            int vertex = vertices[j];
            omp_set_lock(&locks[vertex]);
            m_objGradient.block(3*vertex,0,3,m_objNum) -= tmp.block(3*j,0,3,m_objNum);
            omp_unset_lock(&locks[vertex]);
        }

        #pragma omp atomic
        ++progressMonitor;

        if(prt==0) {
          int nTicksToPrint = int(progressFactor*progressMonitor)-nTicksPrinted;
          if( nTicksToPrint>0) {
            for(int k=0; k<nTicksToPrint; ++k) cout << "-";
            nTicksPrinted+=nTicksToPrint;
            cout << std::flush;
          }
        }
    }
    // denormalize
    m_objGradient *= m_objValues.asDiagonal();

    for(int i=0; i<Nv; ++i) omp_destroy_lock(&locks[i]);

    for(int k=0; k<75-nTicksPrinted; ++k) cout << "-";
    cout << "|       |" << endl;
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    m_tGeoJac += omp_get_wtime();
    return 0;
}

int PressureBasedCoupledSolverAdjoint::m_allocateFlowJacobian()
{
    m_tFlwJac -= omp_get_wtime();
    int Nc = m_inData_p->cellNumber;
    ArrayXi sizes(Nc);

    // color the rows of the Jacobian to perform the vector-matrix
    // multiplication without reduction variables
    vector<int8_t> cellColor(Nc,-1);
    vector<int> colorSize(1,0);
    int color, nColor=1;
    constexpr int groupSize = m_flowJacobian.color_groupSize;

    {
        // for each color keep track of the column indices that are in it
        vector<vector<bool> > cellInColor(1,vector<bool>(Nc,false));

        for(int cell=0; cell<Nc; cell+=groupSize)
        {
            int localSize = min(groupSize,Nc-cell);

            vector<vector<int> > cells(localSize);

            //#pragma omp parallel for schedule(dynamic,1)
            for(int j=0; j<localSize; ++j)
            {
                vector<int> vertices, faces, faceDegEndIdx, cellDegEngIdx;
                m_gatherNeighbourhood<true>(cell+j,2,vertices,faces,faceDegEndIdx,cells[j],cellDegEngIdx);
                sizes(cell+j) = cells[j].size();
            }

            for(color=0; color<nColor; ++color)
            {
                bool free = true;
                for(int j=0; j<localSize && free; ++j)
                    for(auto i : cells[j])
                        free &= !cellInColor[color][i];
                if(free) break;
            }

            // no color was free, make space for a new one
            if(color==nColor)
            {
                ++nColor;
                colorSize.push_back(0);
                cellInColor.push_back(vector<bool>(Nc,false));
            }

            // assign color, update its count, mark the column indices
            colorSize[color] += localSize;
            for(int j=0; j<localSize; ++j)
            {
                cellColor[cell+j] = color;
                for(auto i : cells[j]) cellInColor[color][i] = true;
            }
        }
    }

    // convert coloring into a CSR storage system
    m_flowJacobian.color_outerIdxPtr.resize(nColor+1,0);
    m_flowJacobian.color_cellIdx.reserve(Nc);

    for(color=0; color<nColor; ++color)
    {
        m_flowJacobian.color_outerIdxPtr[color+1] =
            m_flowJacobian.color_outerIdxPtr[color] + colorSize[color];

        for(int cell=0; cell<Nc; cell+=groupSize)
            if(cellColor[cell]==color)
                for(int j=0; j<groupSize && (cell+j)<Nc; ++j)
                    m_flowJacobian.color_cellIdx.push_back(cell+j);
    }

    // allocate and initialize BCSR structures
    int outerSize = Nc+1,
        innerSize = sizes.sum();

    m_flowJacobian.outerIdxPtr.resize(outerSize);
    m_flowJacobian.cellIdx.resize(innerSize);
    m_flowJacobian.coefficients.resize(4*innerSize,NoChange);

    m_flowJacobian.product.resize(4*Nc,m_objNum);

    m_flowJacobian.outerIdxPtr(0) = 0;
    for(int i=0; i<Nc; ++i)
        m_flowJacobian.outerIdxPtr(i+1) = m_flowJacobian.outerIdxPtr(i)+sizes(i);

    assert(m_flowJacobian.outerIdxPtr(Nc) == innerSize);
    m_tFlwJac += omp_get_wtime();
    return 0;
}

template<bool cellsOnly>
int PressureBasedCoupledSolverAdjoint::m_gatherNeighbourhood(const int cell,
                                                             const int maxDegree,
                                                             vector<int>& vertices,
                                                             vector<int>& faces,
                                                             vector<int>& faceDegEndIdx,
                                                             vector<int>& cells,
                                                             vector<int>& cellDegEndIdx) const
{
    using mathUtils::matrix::existsInVector;
    if(!cellsOnly) {
    vertices.clear(); faces.clear(); faceDegEndIdx.clear(); cellDegEndIdx.clear();
    faceDegEndIdx.reserve(maxDegree+1); cellDegEndIdx.reserve(maxDegree+1);
    }
    cells.clear();
    // there are cell(face)DegEndIdx[i]-cell(face)DegEndIdx[i-1] cells(faces) of degree i

    // Epicenter of search
    cells.push_back(cell);
    if(!cellsOnly) cellDegEndIdx.push_back(1);

    // Neighbours
    int Ni = 0, Nf;
    for(int deg=1; deg<=maxDegree; ++deg)
    {
        Nf = cells.size();
        for(int i=Ni; i<Nf; ++i)
        {
            int C0  = cells[i];
            for(int Fj : m_connectivity_F[C0])
            {
                if(!cellsOnly)
                if(!existsInVector(faces,Fj))
                    faces.push_back(Fj);

                int C1 = (m_inData_p->connectivity[Fj].first == C0)?
                    m_inData_p->connectivity[Fj].second : m_inData_p->connectivity[Fj].first;

                if(C1 == -1)
                    if(m_inData_p->boundaryType(Fj) == BoundaryType::PERIODIC)
                        C1 = m_inData_p->boundaryConditions(Fj,0);

                if(C1!=-1 && !existsInVector(cells,C1))
                    cells.push_back(C1);
            }
        }
        if(!cellsOnly) {
        faceDegEndIdx.push_back(faces.size());
        cellDegEndIdx.push_back(cells.size());}
        Ni = Nf;
    }

    if(cellsOnly) return 0;

    // Add faces for last degree (outer faces)
    Nf = cells.size();
    for(int i=Ni; i<Nf; ++i)
    {
        int C0  = cells[i];
        for(int Fj : m_connectivity_F[C0])
            if(!existsInVector(faces,Fj))
                faces.push_back(Fj);
    }
    faceDegEndIdx.push_back(faces.size());

    // Get vertices for faces
    for(int Fj : faces)
        for(int i=m_inData_p->verticesStart[Fj];
                i<m_inData_p->verticesStart[Fj+1]; ++i)
        {
            int vertex = m_inData_p->verticesIndex[i];
            if(!existsInVector(vertices,vertex)) vertices.push_back(vertex);
        }

    return 0;
}
template int PressureBasedCoupledSolverAdjoint::m_gatherNeighbourhood<false>(const int,
    const int, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&) const;
template int PressureBasedCoupledSolverAdjoint::m_gatherNeighbourhood<true>(const int,
    const int, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&) const;

void PressureBasedCoupledSolverAdjoint::flowJacobianMatrixType::gemm(const int nCols, const numPrec* mat)
{
    timer -= omp_get_wtime();
    int Nc = outerIdxPtr.rows()-1;

    for(size_t color=1; color<color_outerIdxPtr.size(); ++color)
    {
        // within each color run in parallel in chunks multiple of the group size parameter
        #pragma omp parallel for schedule(static,10*color_groupSize)
        for(int rptr=color_outerIdxPtr[color-1]; rptr<color_outerIdxPtr[color]; ++rptr)
        {
            int i = color_cellIdx[rptr]; // the row index

            int nbrPtr = outerIdxPtr(i),
                nbrNum = outerIdxPtr(i+1)-outerIdxPtr(i);

            Matrix<numPrec,4,Dynamic> fetch(4,nCols);
            for(int j=0; j<nCols; ++j)
              for(int k=0; k<4; ++k)
                fetch(k,j) = mat[4*Nc*j+4*i+k];

            Matrix<numPrec,Dynamic,Dynamic> tmp = coefficients.block(4*nbrPtr,0,4*nbrNum,4)*fetch;

            for(int j=0; j<nbrNum; ++j) {
                int idx = cellIdx(nbrPtr+j);
                product.block(4*idx,0,4,nCols) += tmp.block(4*j,0,4,nCols);
            }
        }
    }

    timer += omp_get_wtime();
}

void PressureBasedCoupledSolverAdjoint::flowJacobianMatrixType::free()
{
    outerIdxPtr.resize(0);    cellIdx.resize(0);
    coefficients.resize(0,4); product.resize(0,0);
    std::vector<int>().swap(color_cellIdx);
}

void PressureBasedCoupledSolverAdjoint::flowJacobianMatrixType::writeToFile(const std::string &filePath)
{
    std::ofstream file;
    file.open(filePath.c_str(),std::ios::binary);

    #define WRITE4B(p) file.write(reinterpret_cast<char *>(&(p)),4)

    int Nc = outerIdxPtr.rows()-1, nnz = cellIdx.rows()*16, o=0;

    WRITE4B(Nc); WRITE4B(nnz); WRITE4B(o);

    for(int i=0; i<Nc; ++i)
      for(int k=outerIdxPtr(i); k<outerIdxPtr(i+1); ++k)
        for(int ii=0; ii<4; ++ii)
          for(int jj=0; jj<4; ++jj)
          {
              float val = coefficients(4*k+jj,ii);
              if(std::abs(val)>0.0f)
              {
                  int row = 4*i+ii;
                  int col = 4*cellIdx(k)+jj;

                  WRITE4B(row);  WRITE4B(col);  WRITE4B(val);
              }
          }
    file.close();
    #undef WRITE4B
}
}
