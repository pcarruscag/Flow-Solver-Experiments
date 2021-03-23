//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow.h"

#include <fstream>
#include <string>

namespace flow
{
int PressureBasedCoupledSolver::initialize(const std::string &filePath)
{
    if(!m_meshIsSet)
        return 1;
    if(!m_bndrsSet)
        return 2;
    if(m_allocate()!=0)
        return 3;

    std::ifstream file;
    file.open(filePath.c_str(),std::ios::binary);
    if(!file.good())
        return 4;

    char data[4];
    file.read(data,4); // consume number of partitions in file

    int nCellMesh = 0, nCellFile = 0, cursor = 0;

    for(int prt=0; prt<m_partNum; ++prt)
    {
        nCellMesh += (*m_part)[prt].cells.number;

        for(int i=0; i<(*m_part)[prt].cells.number && !file.eof(); ++i)
        {
            if(cursor == nCellFile) { // consume the partition size
                file.read(data,4);
                nCellFile += *reinterpret_cast<int*>(data);
            }
            file.read(data,4); m_flwFld_C[prt].u(i) = *reinterpret_cast<float*>(data);
            file.read(data,4); m_flwFld_C[prt].v(i) = *reinterpret_cast<float*>(data);
            file.read(data,4); m_flwFld_C[prt].w(i) = *reinterpret_cast<float*>(data);
            file.read(data,4); m_flwFld_C[prt].p(i) = *reinterpret_cast<float*>(data);
            file.read(data,4); m_flwFld_C[prt].turb1(i) = *reinterpret_cast<float*>(data);
            file.read(data,4); m_flwFld_C[prt].turb2(i) = *reinterpret_cast<float*>(data);
            ++cursor;
        }
    }
    file.close();

    if(nCellFile!=nCellMesh) // file too small or too big
        return 5;

    m_isInit = true;
    return 0;
}

int PressureBasedCoupledSolver::getSolution(const UnstructuredMesh& coalescedMesh,
                                            MatrixXf* solution_p) const
{
    if(!m_finished) return 1;
    if(coalescedMesh.m_originFace.size() != size_t(m_partNum)) return 2;

    // Determine the vertex values on the coalesced mesh (before partitioning)
    //   Done via inverse distance weighted averaging from face centers to vertices.
    //   We assume only the connectivity of the coalesced mesh is available,
    //   not its geometric properties.
    //   To do this in parallel we need to employ a reduction strategy because
    //   the same vertex is shared by many faces.

    std::vector<MatrixXf> flwFld_V(m_partNum);
    std::vector<VectorXf> sumOfWeights(m_partNum);
    int Nv = coalescedMesh.vertices.number;

    #pragma omp parallel num_threads(m_partNum)
    {
        int prt = omp_get_thread_num();
        int Nfl = coalescedMesh.m_originFace[prt].size(),
            Nbl = m_boundaries[prt].number;

        assert(Nfl == (*m_part)[prt].faces.number);

        flwFld_V[prt].setZero(Nv,6);
        sumOfWeights[prt].setZero(Nv);

        // Interpolation from faces to vertices
        for(int f_loc=0; f_loc<Nfl; ++f_loc) {
            int f_glb=coalescedMesh.m_originFace[prt][f_loc];

            float bndFactor = 1.0;
            if(f_loc<Nbl)
              if(m_boundaries[prt].type(f_loc) < BoundaryType::SYMMETRY)
                bndFactor = 1000.0;

            int NfaceVerts = coalescedMesh.faces.verticesStart[f_glb+1]-
                             coalescedMesh.faces.verticesStart[f_glb];

            for(int i=0; i<NfaceVerts; ++i)
            {
                int v_loc_p = (*m_part)[prt].faces.verticesStart[f_loc]+i,
                    v_glb_p =  coalescedMesh.faces.verticesStart[f_glb]+i;

                int v_loc = (*m_part)[prt].faces.verticesIndex[v_loc_p],
                    v_glb =  coalescedMesh.faces.verticesIndex[v_glb_p];

                float w = bndFactor/((*m_part)[prt].faces.centroid.row(f_loc)-
                                     (*m_part)[prt].vertices.coords.row(v_loc)).norm();

                sumOfWeights[prt](v_glb) += w;
                flwFld_V[prt](v_glb,0) += w*m_flwFld_F[prt].u(f_loc);
                flwFld_V[prt](v_glb,1) += w*m_flwFld_F[prt].v(f_loc);
                flwFld_V[prt](v_glb,2) += w*m_flwFld_F[prt].w(f_loc);
                flwFld_V[prt](v_glb,3) += w*m_flwFld_F[prt].p(f_loc);
                flwFld_V[prt](v_glb,4) += w*m_flwFld_F[prt].turb1(f_loc);
                flwFld_V[prt](v_glb,5) += w*m_flwFld_F[prt].turb2(f_loc);
            }
        }
        // Reduction
        #pragma omp barrier
        ArrayXi tmp;
        tmp.setLinSpaced(m_partNum+1,0,Nv);
        int start = tmp(prt),
            numel = tmp(prt+1)-tmp(prt);

        for(int i=1; i<m_partNum; ++i) {
            flwFld_V[0].block(start,0,numel,6) += flwFld_V[i].block(start,0,numel,6);
            sumOfWeights[0].segment(start,numel) += sumOfWeights[i].segment(start,numel);
        }
    }
    // Average
    solution_p->resize(Nv,6);

    for(int i=0; i<Nv; ++i)
        solution_p->row(i) = flwFld_V[0].row(i)/sumOfWeights[0](i);

    return 0;
}

int PressureBasedCoupledSolver::saveState(const std::string &filePath) const
{
    if(!m_finished) return 1;

    std::ofstream file;
    file.open(filePath.c_str(),std::ios::binary);
    if(!file.is_open()) return 2;

    #define WRITE(T,V) {T _v = V; file.write(reinterpret_cast<char *>(&_v),4);}
    WRITE(int,m_partNum)
    for(int prt=0; prt<m_partNum; ++prt)
    {
        WRITE(int,(*m_part)[prt].cells.number)
        for(int i=0; i<(*m_part)[prt].cells.number; ++i)
        {
            WRITE(float,m_flwFld_C[prt].u(i))
            WRITE(float,m_flwFld_C[prt].v(i))
            WRITE(float,m_flwFld_C[prt].w(i))
            WRITE(float,m_flwFld_C[prt].p(i))
            WRITE(float,m_flwFld_C[prt].turb1(i))
            WRITE(float,m_flwFld_C[prt].turb2(i))
        }
    }
    file.close();
    #undef WRITE
    return 0;
}

int PressureBasedCoupledSolver::getAdjointInput(const UnstructuredMesh& coalescedMesh,
                                                AdjointInputData* data)
{
    if(!m_finished)
        return 1;

    // Boundary and face count
    data->boundaryNumber = 0;  data->faceNumber = 0;

    for(int prt=0; prt<m_partNum; ++prt)
    {
        data->boundaryNumber +=  m_boundaries[prt].number-m_ghostCells[prt].number;
        data->faceNumber += 2*(*m_part)[prt].faces.number-m_ghostCells[prt].number;
    }
    data->faceNumber /= 2;

    // Properties and domain conditions
    data->rho = m_rho;
    data->mu  = m_mu;
    data->rotationalSpeed = m_rotationalSpeed;
    data->venkatK = m_control.venkatK;

    // Cell center values
    data->cellNumber = 0;
    ArrayXi cellOffset; // from local to unpartitioned numbering
    data->u.resize(0);  data->v.resize(0);   data->w.resize(0);
    data->p.resize(0);  data->mut.resize(0); data->residuals.resize(0);
    data->ulim.resize(0); data->vlim.resize(0); data->wlim.resize(0);
    data->mdiag.resize(0);

    cellOffset.setZero(m_partNum);
    for(int prt=0; prt<m_partNum; ++prt)
    {
        int N = 4*(*m_part)[prt].cells.number;
        cellOffset(prt) = data->residuals.rows();
        int newSize = cellOffset(prt)+N;
        data->residuals.conservativeResize(newSize);

        for(int i=0; i<N; ++i)
            data->residuals(cellOffset(prt)+i) = m_solution[prt](i);
    }
    cellOffset.setZero(m_partNum);
    for(int prt=0; prt<m_partNum; ++prt)
    {
        int Nc = (*m_part)[prt].cells.number;
        data->cellNumber += Nc;
        cellOffset(prt) = data->u.rows();
        int newSize = cellOffset(prt)+Nc;
        data->u.conservativeResize(newSize);
        data->v.conservativeResize(newSize);
        data->w.conservativeResize(newSize);
        data->p.conservativeResize(newSize);
        data->mut.conservativeResize(newSize);
        data->ulim.conservativeResize(newSize);
        data->vlim.conservativeResize(newSize);
        data->wlim.conservativeResize(newSize);
        data->mdiag.conservativeResize(newSize);

        for(int i=0; i<Nc; ++i)
        {
            data->u(cellOffset(prt)+i) = m_flwFld_C[prt].u(i);
            data->v(cellOffset(prt)+i) = m_flwFld_C[prt].v(i);
            data->w(cellOffset(prt)+i) = m_flwFld_C[prt].w(i);
            data->p(cellOffset(prt)+i) = m_flwFld_C[prt].p(i);
            data->mut(cellOffset(prt)+i) = m_mut[prt](i);
            data->ulim(cellOffset(prt)+i) = m_limit[prt].u(i);
            data->vlim(cellOffset(prt)+i) = m_limit[prt].v(i);
            data->wlim(cellOffset(prt)+i) = m_limit[prt].w(i);
            data->mdiag(cellOffset(prt)+i) = m_source_t[prt](i);
        }
    }
    assert(data->cellNumber == data->u.rows());

    // Get the mesh definition from the un-partitioned domain
    data->vertexNumber  = coalescedMesh.vertices.number;
    data->verticesCoord = coalescedMesh.vertices.coords;
    data->connectivity  = coalescedMesh.faces.connectivity;
    data->verticesStart = coalescedMesh.faces.verticesStart;
    data->verticesIndex = coalescedMesh.faces.verticesIndex;

    // setup coalesced boundaries using face map created during partitioning
    data->boundaryType.resize(data->boundaryNumber);
    data->boundaryConditions.resize(data->boundaryNumber,6);

    #pragma omp parallel num_threads(m_partNum)
    {
        int prt = omp_get_thread_num();

        for(int i=0; i<m_boundaries[prt].number; ++i)
        {
            int bndType = m_boundaries[prt].type(i);
            if(bndType != BoundaryType::GHOST) { // coalesced does not have ghost faces
                int f_glb = coalescedMesh.m_originFace[prt][i];
                data->boundaryType(f_glb) = bndType;
                data->boundaryConditions.row(f_glb) = m_boundaries[prt].conditions.row(i);
                if(bndType == BoundaryType::PERIODIC) {
                    data->boundaryConditions(f_glb,0) += cellOffset(prt);
                    int matchFace = data->boundaryConditions(f_glb,4);
                    data->boundaryConditions(f_glb,4) = coalescedMesh.m_originFace[prt][matchFace];
                }
            }
        }
    }

    // Copy the transpose of the matrix
    data->rowPtr.resize(m_partNum);
    data->colIdx.resize(m_partNum);
    data->Aii.resize(m_partNum);
    data->Aij.resize(m_partNum);
    data->Aij_ptr.resize(m_partNum);
    data->communications = m_communications;

    // first transpose "inplace" (enough for the main blocks)
    #pragma omp parallel num_threads(m_partNum)
    {
        int prt = omp_get_thread_num();

        // copy matrix sparse pattern to local data
        int rows = m_coefMat_t[prt].rows();
        int nnzs = m_coefMat_t[prt].nonZeros();

        data->rowPtr[prt].resize(rows+1);
        data->colIdx[prt].resize(nnzs);

        for(int i=0; i<rows+1; ++i)
          data->rowPtr[prt][i] = m_coefMat_t[prt].outerIndexPtr()[i];

        for(int i=0; i<nnzs; ++i)
          data->colIdx[prt][i] = m_coefMat_t[prt].innerIndexPtr()[i];

        // copy matrix coeffs, reseat pattern pointers and transpose
        data->Aii[prt] = m_coefMat[prt];
        data->Aii[prt].setStructure(rows,nnzs,data->rowPtr[prt].data(),
                                    data->colIdx[prt].data());
        data->Aii[prt].transposeInPlace();

        // copy ghost blocks transposing them
        int N = m_ghostCells[prt].number;
        data->Aij[prt].resize(N);
        for(int i=0; i<N; ++i)
            data->Aij[prt][i] = m_ghostCells[prt].coefMat[i].cast<AdjointInputData::solvePrec>().transpose();

        data->Aij_ptr[prt] = m_ghostCells[prt].nghbrCell;
    }

    // flow solver matrices no longer needed
    std::vector<SBCSRmatrix<float> >().swap(m_coefMat);
    std::vector<SparseMatrix<float,RowMajor> >().swap(m_coefMat_t);

    // now we need to swap the Aij's
    for(auto c : m_communications)
        for(int i=0; i<c.length; ++i)
        {
            int idxI = c.startI+i,
                idxJ = c.startJ+i;
            Matrix<AdjointInputData::solvePrec,4,4> temp  = data->Aij[c.partI][idxI];
            data->Aij[c.partI][idxI] = data->Aij[c.partJ][idxJ];
            data->Aij[c.partJ][idxJ] = temp;
        }

    return 0;
}

}
