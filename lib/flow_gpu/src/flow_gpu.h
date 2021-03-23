//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef FLOW_GPU_H
#define FLOW_GPU_H

// Experiment: Port the shared memory implementation of SIMPLE to OpenCL
// Warning: Shameless use of macros to automate some OpenCL boilerplate

#include "../../mesh/src/unstructuredMesh.h"
#include "../../fileManagement/src/flowParameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <CL/cl.h>

using namespace fileManagement;
using namespace mesh;
using namespace Eigen;

namespace flow
{
    class PressureBasedSegregatedSolver
    {
      private:
        // integer scalar, vector, and matrix types
        typedef cl_int int_t;
        typedef Matrix<int_t,Dynamic,1> vectorXi_t;
        typedef Matrix<int_t,Dynamic,Dynamic> matrixXi_t;

        // floating-point scalar, vector, and matrix types
        typedef cl_float float_t;
        typedef Matrix<float_t,Dynamic,1> vectorXf_t;
        typedef Matrix<float_t,Dynamic,Dynamic> matrixXf_t;

        // structs to group important variables by type
        struct adjacency
        {
            // adjacency data, prepared on cpu and transferred
            int_t maxNeighbours;
            vectorXi_t cellNum;
            matrixXi_t cellIdx;
            vectorXi_t faceNum;
            matrixXi_t faceIdx;
            matrixXf_t faceDir;
            matrixXi_t connect;

            // read-only buffers
            cl_uint cl_width;
            cl_mem cl_cellNum;
            cl_mem cl_cellIdx;
            cl_mem cl_faceIdx;
            cl_mem cl_faceDir;
            cl_mem cl_connect;

            adjacency() : cl_cellNum(NULL), cl_cellIdx(NULL),
                          cl_faceIdx(NULL), cl_faceDir(NULL),
                          cl_connect(NULL) {}
            ~adjacency()
            {
                clReleaseMemObject(cl_cellNum);
                clReleaseMemObject(cl_cellIdx);
                clReleaseMemObject(cl_faceIdx);
                clReleaseMemObject(cl_faceDir);
                clReleaseMemObject(cl_connect);
            }
        };

        struct geometry
        {
            // geometric properties of the grid, contained in mesh
            // sizes
            cl_uint cl_nCells;
            cl_uint cl_nFaces;
            // face and cell properties (read-only)
            cl_mem cl_area;
            cl_mem cl_wf;
            cl_mem cl_r0;
            cl_mem cl_r1;
            cl_mem cl_d0;
            cl_mem cl_d1;
            cl_mem cl_a0;
            cl_mem cl_a1;
            cl_mem cl_volume;

            // permanent variables that can be passed to kernels
            // and associate coordinate directions with columns.
            cl_uint cl_xDim;
            cl_uint cl_yDim;
            cl_uint cl_zDim;
            cl_uint cl_nDim;

            geometry() : cl_area(NULL), cl_wf(NULL), cl_r0(NULL), cl_r1(NULL),
                         cl_d0(NULL), cl_d1(NULL), cl_a0(NULL), cl_a1(NULL),
                         cl_volume(NULL), cl_xDim(0), cl_yDim(1), cl_zDim(2),
                         cl_nDim(3) {}
            ~geometry()
            {
                clReleaseMemObject(cl_area);
                clReleaseMemObject(cl_wf);
                clReleaseMemObject(cl_r0);
                clReleaseMemObject(cl_r1);
                clReleaseMemObject(cl_d0);
                clReleaseMemObject(cl_d1);
                clReleaseMemObject(cl_a0);
                clReleaseMemObject(cl_a1);
                clReleaseMemObject(cl_volume);
            }
        };

        struct flowField
        {
            // conservatives at cell centroids and face mass flows
            cl_mem cl_u;
            cl_mem cl_v;
            cl_mem cl_w;
            cl_mem cl_p;
            cl_mem cl_mdot;

            // permanent variables that can be passed to kernels
            // and associate variables with column indexes.
            cl_uint cl_uVar;
            cl_uint cl_vVar;
            cl_uint cl_wVar;
            cl_uint cl_pVar;
            cl_uint cl_pcVar;

            flowField() : cl_u(NULL), cl_v(NULL), cl_w(NULL), cl_p(NULL),
                          cl_mdot(NULL),
                          cl_uVar(0), cl_vVar(1), cl_wVar(2), cl_pVar(3),
                          cl_pcVar(4) {}
            ~flowField()
            {
                clReleaseMemObject(cl_u);
                clReleaseMemObject(cl_v);
                clReleaseMemObject(cl_w);
                clReleaseMemObject(cl_p);
                clReleaseMemObject(cl_mdot);
            }
        };

        struct boundaries
        {
            int number;
            vectorXi_t type;
            matrixXf_t conditions;

            cl_uint cl_number;
            cl_mem cl_type;
            // a sub-set of conditions, for given value boundaries
            // contains fixed values set at initialization, for other
            // boundaries values are computed every iteration.
            cl_mem cl_values;

            boundaries() : cl_type(NULL), cl_values(NULL) {}

            ~boundaries()
            {
                clReleaseMemObject(cl_type);
                clReleaseMemObject(cl_values);
            }
        };

        struct gradLimit
        {
            cl_mem  cl_gradP; // gradient of pressure or its correction
            cl_mem* cl_gradU; // gradient of a velocity component
            cl_mem* cl_limit; // limiter
            // associated intermediate variables
            cl_mem* cl_phiF;  // values interpolated at faces
            cl_mem* cl_flux;  // Ai*phiF
            cl_mem* cl_delta; // phi1-phi0
            cl_mem* cl_proj0; // dot(nabla0, r0)
            cl_mem* cl_proj1; // dot(nabla1, r1)

            gradLimit() : cl_gradP(NULL), cl_gradU(NULL), cl_limit(NULL),
                          cl_phiF(NULL),  cl_flux(NULL),  cl_delta(NULL),
                          cl_proj0(NULL), cl_proj1(NULL) {}
            ~gradLimit()
            {
                clReleaseMemObject(cl_gradP);
            }
        };

        struct linearSys
        {
            // coefficients of the system matrix and the pressure
            // correction preconditioner (SPAI), indices are in cellIdx
            cl_mem  cl_offDiag;
            cl_mem* cl_offDiagSPAI;
            cl_mem  cl_diagonal;
            cl_mem* cl_diagonalSPAI;

            // matrix construction variables
            cl_mem  cl_diffCoeff;
            cl_mem* cl_faceType;
            cl_mem* cl_corrFlux;
            cl_mem* cl_diagCopy;

            // right and left hand sides
            cl_mem* cl_source;
            cl_mem* cl_solution;

            // partially reduced dot product, used in Krylov methods
            vectorXf_t partDot;
            cl_mem  cl_partDot;
            size_t sizePartDot;

            linearSys() : cl_offDiag(NULL),  cl_offDiagSPAI(NULL),
                          cl_diagonal(NULL), cl_diagonalSPAI(NULL),
                          cl_diffCoeff(NULL),cl_faceType(NULL),
                          cl_corrFlux(NULL), cl_diagCopy(NULL),
                          cl_source(NULL),   cl_solution(NULL) {}
            ~linearSys()
            {
                clReleaseMemObject(cl_offDiag);
                clReleaseMemObject(cl_diagonal);
                clReleaseMemObject(cl_diffCoeff);
                clReleaseMemObject(cl_partDot);
            }
        };

        struct linearSol
        {
            // working vectors used by Krylov methods
            cl_mem cl_v;
            cl_mem cl_p;
            cl_mem* cl_y;
            cl_mem* cl_z;
            cl_mem* cl_r;
            cl_mem* cl_r0;

            linearSol() : cl_v(NULL), cl_p(NULL), cl_y(NULL),
                          cl_z(NULL), cl_r(NULL), cl_r0(NULL) {}
            ~linearSol()
            {
                clReleaseMemObject(cl_v);
                clReleaseMemObject(cl_p);
            }
        };

        struct memoryPool
        {
            // working/auxilary buffers used in more than one operation.
            cl_mem cl_cellBuf1; // Nc*1
            cl_mem cl_faceBuf1; // Nf*1
            cl_mem cl_faceBuf2; // Nf*1
            cl_mem cl_faceBuf3; // Nf*1
            cl_mem cl_largeBuf; // Nc*width

            memoryPool() : cl_cellBuf1(NULL), cl_faceBuf1(NULL),
                           cl_faceBuf2(NULL), cl_faceBuf3(NULL),
                           cl_largeBuf(NULL) {}
            ~memoryPool()
            {
                clReleaseMemObject(cl_cellBuf1);
                clReleaseMemObject(cl_faceBuf1);
                clReleaseMemObject(cl_faceBuf2);
                clReleaseMemObject(cl_faceBuf3);
                clReleaseMemObject(cl_largeBuf);
            }
        };

      public:
        // public methods
        PressureBasedSegregatedSolver();
        ~PressureBasedSegregatedSolver();
        void setControl(const FlowParamManager& flowParams);
        int setMesh(UnstructuredMesh* domain);
        int applyBndConds(const FlowParamManager& flowParams);
        int run();
        int saveState(const std::string &filePath) const;
        int getSolution(MatrixXf* solution_p) const;

      private:
        // preprocessing methods
        void m_buildAdjacency(); // generate adjacency from the connectivity (graph)
        void m_allocate(); // create device buffers and initialize them
        void m_defaultInit();

        // algorithmic steps, wrap the enqueueing of kernels
        void m_boundaryValues();
        void m_assembleMomMat();
        void m_faceValues(const cl_uint* var);
        void m_computeGradient(const cl_uint* var);
        void m_computeLimiter(const cl_uint* var);
        void m_compute2ndOrderFluxes(const cl_uint* var);
        void m_computeMomSource(const cl_uint* var);
        void m_solveMomentum(const cl_uint* var);
        void m_assemblePCmat();
        void m_computeMassFlows();
        void m_solvePressureCorrection();
        void m_applyCorrections();

        // OpenCL environment objects
        cl_command_queue m_cmdQueue;
        cl_platform_id   m_platform;
        cl_device_id     m_device;
        cl_context       m_context;
        cl_program       m_program;
        cl_event         m_event;

        // OpenCL kernels
        // algorithmic
        cl_kernel m_ckBoundaryValues;
        cl_kernel m_ckFaceValues;
        cl_kernel m_ckComputeGradient1;
        cl_kernel m_ckComputeGradient2;
        cl_kernel m_ckComputeLimiter1;
        cl_kernel m_ckComputeLimiter2;
        cl_kernel m_ckAssembleMomMat1;
        cl_kernel m_ckAssembleMomMat2;
        cl_kernel m_ckAssembleMomMat3;
        cl_kernel m_ckComp2ndOrderFlux;
        cl_kernel m_ckComputeMomSrc1;
        cl_kernel m_ckAssemblePCmat1;
        cl_kernel m_ckAssemblePCmat2;
        cl_kernel m_ckAssemblePCmat3;
        cl_kernel m_ckComputeMassFlows;
        cl_kernel m_ckCorrectMassFlows;
        cl_kernel m_ckBoundaryValuesPC;
        cl_kernel m_ckCorrectVelocity;
        // linear algebra
        cl_kernel m_ckSET;
        cl_kernel m_ckCOPY;
        cl_kernel m_ckAXPY;
        cl_kernel m_ckDOT;
        cl_kernel m_ckGEMV;
        cl_kernel m_ckGBMV;
        cl_kernel m_ckSCL;
        cl_kernel m_ckSPAI;

        // OpenCL work sizes, local and global for each kind of loop
        size_t m_localSize;
        size_t m_glbSizeCells;
        size_t m_glbSizeFaces;
        size_t m_glbSizeBndrs;

        // flags to determine what part of the process we are on
        bool m_ctrlIsSet;
        bool m_meshIsSet;
        bool m_bndrsSet;
        bool m_isInit;
        bool m_finished;

        fileManagement::FlowParamManager::controlParameters m_control;
        UnstructuredMesh* m_mesh;
        matrixXf_t m_solution;

        float_t m_rho;
        float_t m_mu;
        float_t m_relaxV;
        float_t m_relaxP;
        float_t m_minusOne;
        float_t m_plusOne;
        float_t m_zero;
        float_t m_const;

        adjacency  m_adjacency;
        geometry   m_geometry;
        flowField  m_field;
        boundaries m_boundaries;
        gradLimit  m_gradLimit;
        linearSys  m_linearSys;
        linearSol  m_linearSol;
        memoryPool m_memoryPool;

        Array<float_t,1,4> m_res;
        Array<float_t,1,4> m_resNorm;

        // wrap calls to linear algebra kernels used in many places

        inline cl_int m_blas_SET(const size_t& N,
                                 const cl_uint& vecSz,
                                 const float_t& val,
                                 const cl_mem* vec) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckSET, 0, sizeof(cl_uint), (void*)&vecSz);
            errNum |= clSetKernelArg(m_ckSET, 1, sizeof(float_t), (void*)&val);
            errNum |= clSetKernelArg(m_ckSET, 2, sizeof(cl_mem),  (void*)vec);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckSET,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }

        inline cl_int m_blas_COPY(const size_t& N,
                                  const cl_uint& vecSz,
                                  const cl_mem* src,
                                  const cl_mem* dst) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckCOPY, 0, sizeof(cl_uint), (void*)&vecSz);
            errNum |= clSetKernelArg(m_ckCOPY, 1, sizeof(cl_mem),  (void*)src);
            errNum |= clSetKernelArg(m_ckCOPY, 2, sizeof(cl_mem),  (void*)dst);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckCOPY,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }

        inline cl_int m_blas_AXPY(const size_t& N,
                                  const cl_uint& vecSz,
                                  const float_t& alpha,
                                  const cl_mem* x,
                                  const cl_mem* y) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckAXPY, 0, sizeof(cl_uint), (void*)&vecSz);
            errNum |= clSetKernelArg(m_ckAXPY, 1, sizeof(float_t), (void*)&alpha);
            errNum |= clSetKernelArg(m_ckAXPY, 2, sizeof(cl_mem),  (void*)x);
            errNum |= clSetKernelArg(m_ckAXPY, 3, sizeof(cl_mem),  (void*)y);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAXPY,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }

        inline cl_int m_blas_DOT(const cl_mem* u,
                                 const cl_mem* v,
                                 float_t& res)
        {
            // the vectors are assumed to be nCells long as "partDot" is sized accordingly
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckDOT, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
            errNum |= clSetKernelArg(m_ckDOT, 1, sizeof(cl_mem),  (void*)u);
            errNum |= clSetKernelArg(m_ckDOT, 2, sizeof(cl_mem),  (void*)v);
            errNum |= clSetKernelArg(m_ckDOT, 3, sizeof(cl_mem),  (void*)&m_linearSys.cl_partDot);
            errNum |= clSetKernelArg(m_ckDOT, 4, m_linearSys.sizePartDot, NULL);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckDOT,
                     1, NULL, &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

            errNum = clEnqueueReadBuffer(m_cmdQueue, m_linearSys.cl_partDot, CL_TRUE, 0,
                     m_linearSys.sizePartDot, m_linearSys.partDot.data(), 0, NULL, NULL);

            res = m_linearSys.partDot.sum();

            return errNum;
        }

        inline cl_int m_blas_GEMV(const size_t& N,
                                  const cl_uint& rows,
                                  const cl_uint& width,
                                  const float_t& alpha,
                                  const float_t& beta,
                                  const cl_mem* acols,
                                  const cl_mem* avals,
                                  const cl_mem* x,
                                  const cl_mem* y) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckGEMV, 0, sizeof(cl_uint), (void*)&rows);
            errNum |= clSetKernelArg(m_ckGEMV, 1, sizeof(cl_uint), (void*)&width);
            errNum |= clSetKernelArg(m_ckGEMV, 2, sizeof(float_t), (void*)&alpha);
            errNum |= clSetKernelArg(m_ckGEMV, 3, sizeof(float_t), (void*)&beta);
            errNum |= clSetKernelArg(m_ckGEMV, 4, sizeof(cl_mem),  (void*)acols);
            errNum |= clSetKernelArg(m_ckGEMV, 5, sizeof(cl_mem),  (void*)avals);
            errNum |= clSetKernelArg(m_ckGEMV, 6, sizeof(cl_mem),  (void*)x);
            errNum |= clSetKernelArg(m_ckGEMV, 7, sizeof(cl_mem),  (void*)y);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckGEMV,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }

        inline cl_int m_blas_GBMV(const size_t& N,
                                  const cl_uint& rows,
                                  const float_t& alpha,
                                  const float_t& beta,
                                  const cl_mem* diag,
                                  const cl_mem* x,
                                  const cl_mem* y) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckGBMV, 0, sizeof(cl_uint), (void*)&rows);
            errNum |= clSetKernelArg(m_ckGBMV, 1, sizeof(float_t), (void*)&alpha);
            errNum |= clSetKernelArg(m_ckGBMV, 2, sizeof(float_t), (void*)&beta);
            errNum |= clSetKernelArg(m_ckGBMV, 3, sizeof(cl_mem),  (void*)diag);
            errNum |= clSetKernelArg(m_ckGBMV, 4, sizeof(cl_mem),  (void*)x);
            errNum |= clSetKernelArg(m_ckGBMV, 5, sizeof(cl_mem),  (void*)y);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckGBMV,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }

        inline cl_int m_blas_SCL(const size_t& N,
                                 const cl_uint& rows,
                                 const cl_uint& cols,
                                 const cl_mem* scales,
                                 const cl_mem* mat) const
        {
            cl_int errNum = CL_SUCCESS;

            errNum |= clSetKernelArg(m_ckSCL, 0, sizeof(cl_uint), (void*)&rows);
            errNum |= clSetKernelArg(m_ckSCL, 1, sizeof(cl_uint), (void*)&cols);
            errNum |= clSetKernelArg(m_ckSCL, 2, sizeof(cl_mem),  (void*)scales);
            errNum |= clSetKernelArg(m_ckSCL, 3, sizeof(cl_mem),  (void*)mat);

            if(errNum==CL_SUCCESS)
            errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckSCL,
                     1, NULL, &N, &m_localSize, 0, NULL, NULL);
            return errNum;
        }
    };
}
#endif // FLOW_GPU_H
