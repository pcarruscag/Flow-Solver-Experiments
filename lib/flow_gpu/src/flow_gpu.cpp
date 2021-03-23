//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flow_gpu.h"

#include "../../mathUtils/src/matrix.h"

#include <stdexcept>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <limits>

using namespace mathUtils;

namespace flow
{
PressureBasedSegregatedSolver::PressureBasedSegregatedSolver()
{
    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;

    m_ctrlIsSet = false;
    m_meshIsSet = false;
    m_bndrsSet  = false;
    m_isInit    = false;
    m_finished  = false;

    cout << endl << "### Preparing the OpenCL Environment ###" << endl;

    string platformKey = "NVIDIA";

    m_cmdQueue = NULL;
    m_platform = NULL;
    m_device   = NULL;
    m_context  = NULL;
    m_program  = NULL;
    m_event    = NULL;
    m_localSize= 256;

    m_ckBoundaryValues = NULL;
    m_ckFaceValues      = NULL;
    m_ckComputeGradient1 = NULL;
    m_ckComputeGradient2 = NULL;
    m_ckComputeLimiter1  = NULL;
    m_ckComputeLimiter2  = NULL;
    m_ckAssembleMomMat1  = NULL;
    m_ckAssembleMomMat2  = NULL;
    m_ckAssembleMomMat3  = NULL;
    m_ckComp2ndOrderFlux = NULL;
    m_ckComputeMomSrc1   = NULL;
    m_ckAssemblePCmat1   = NULL;
    m_ckAssemblePCmat2   = NULL;
    m_ckAssemblePCmat3   = NULL;
    m_ckComputeMassFlows = NULL;
    m_ckCorrectMassFlows = NULL;
    m_ckBoundaryValuesPC = NULL;
    m_ckCorrectVelocity  = NULL;

    m_ckSET  = NULL;
    m_ckCOPY = NULL;
    m_ckAXPY = NULL;
    m_ckDOT  = NULL;
    m_ckGEMV = NULL;
    m_ckGBMV = NULL;
    m_ckSCL  = NULL;
    m_ckSPAI = NULL;

    m_minusOne = -1.0f;
    m_plusOne  =  1.0f;
    m_zero     =  0.0f;

    cl_int errNum;
    size_t chBuffSz = 1024;

    // Get the NVIDIA platform
    {
        cl_uint numPlatforms = 0;
        errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
        if(errNum!=CL_SUCCESS || numPlatforms==0)
            throw std::runtime_error("CL_INVALID_PLATFORM");

        vector<cl_platform_id> platforms(numPlatforms);
        errNum = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
        if(errNum!=CL_SUCCESS)
            throw std::runtime_error("CL_INVALID_PLATFORM");

        for(cl_uint i=0; i<numPlatforms; ++i)
        {
            char platInfo[chBuffSz];
            errNum = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, chBuffSz, &platInfo, NULL);
            if(errNum!=CL_SUCCESS)
                throw std::runtime_error("CL_INVALID_PLATFORM");

            if(string(platInfo).find(platformKey) != string::npos)
            {
                cout << "Using platform: " << platInfo << endl;
                m_platform = platforms[i];
                break;
            }
        }

        if(m_platform==NULL)
            throw std::runtime_error("CL_INVALID_PLATFORM");
    }

    // Get the first device in the platform
    {
        cl_uint numDevices;
        errNum = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if(errNum!=CL_SUCCESS || numDevices==0)
            throw std::runtime_error("CL_INVALID_DEVICE");

        vector<cl_device_id> devices(numDevices);
        errNum = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), NULL);
        if(errNum!=CL_SUCCESS)
            throw std::runtime_error("CL_INVALID_DEVICE");

        char devInfo[chBuffSz];
        errNum = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, chBuffSz, &devInfo, NULL);
        if(errNum!=CL_SUCCESS)
            throw std::runtime_error("CL_INVALID_DEVICE");

        m_device = devices[0];
        cout << "Using device: " << devInfo << endl;
    }

    m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, &errNum);
    if(errNum!=CL_SUCCESS) throw std::runtime_error("CL_INVALID_CONTEXT");
    cout << "Created context." << endl;

    m_cmdQueue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &errNum);
    if(errNum!=CL_SUCCESS) throw std::runtime_error("CL_INVALID_COMMAND_QUEUE");
    cout << "Created command queue." << endl;

    // Compile the kernel code
    {
        std::ifstream file("flow_gpu.cl");
        if(!file) throw std::runtime_error("CL_INVALID_PROGRAM");

        file.seekg(0, file.end);
        size_t length=file.tellg();
        file.seekg(0, file.beg);

        char* lines = (char*)malloc(length*sizeof(char));
        file.read(lines, length);

        m_program = clCreateProgramWithSource(m_context, 1, (const char **)&lines, &length, &errNum);
        if(errNum!=CL_SUCCESS) {
            free(lines);
            throw std::runtime_error("CL_INVALID_PROGRAM");
        }

        errNum = clBuildProgram(m_program, 1, &m_device, "-cl-fast-relaxed-math", NULL, NULL);
        free(lines);
        if(errNum!=CL_SUCCESS)
        {
            char buildLog[10240];
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buildLog), buildLog, NULL);
            cout << endl << "clBuildProgram failed!" << endl;
            cout << endl << buildLog << endl;
            throw std::runtime_error("CL_INVALID_PROGRAM");
        }
        cout << "Built program." << endl;
    }

    errNum = CL_SUCCESS;
    cl_int flag;
    m_ckBoundaryValues   = clCreateKernel(m_program, "boundaryValues",   &flag); errNum |= flag;
    m_ckFaceValues       = clCreateKernel(m_program, "faceValues",       &flag); errNum |= flag;
    m_ckComputeGradient1 = clCreateKernel(m_program, "computeGradient1", &flag); errNum |= flag;
    m_ckComputeGradient2 = clCreateKernel(m_program, "computeGradient2", &flag); errNum |= flag;
    m_ckComputeLimiter1  = clCreateKernel(m_program, "computeLimiter1",  &flag); errNum |= flag;
    m_ckComputeLimiter2  = clCreateKernel(m_program, "computeLimiter2",  &flag); errNum |= flag;
    m_ckAssembleMomMat1  = clCreateKernel(m_program, "assembleMomMat1",  &flag); errNum |= flag;
    m_ckAssembleMomMat2  = clCreateKernel(m_program, "assembleMomMat2",  &flag); errNum |= flag;
    m_ckAssembleMomMat3  = clCreateKernel(m_program, "assembleMomMat3",  &flag); errNum |= flag;
    m_ckComp2ndOrderFlux = clCreateKernel(m_program, "comp2ndOrderFlux", &flag); errNum |= flag;
    m_ckComputeMomSrc1   = clCreateKernel(m_program, "computeMomSrc1",   &flag); errNum |= flag;
    m_ckAssemblePCmat1   = clCreateKernel(m_program, "assemblePCmat1",   &flag); errNum |= flag;
    m_ckAssemblePCmat2   = clCreateKernel(m_program, "assemblePCmat2",   &flag); errNum |= flag;
    m_ckAssemblePCmat3   = clCreateKernel(m_program, "assemblePCmat3",   &flag); errNum |= flag;
    m_ckComputeMassFlows = clCreateKernel(m_program, "computeMassFlows", &flag); errNum |= flag;
    m_ckCorrectMassFlows = clCreateKernel(m_program, "correctMassFlows", &flag); errNum |= flag;
    m_ckBoundaryValuesPC = clCreateKernel(m_program, "boundaryValuesPC", &flag); errNum |= flag;
    m_ckCorrectVelocity  = clCreateKernel(m_program, "correctVelocity",  &flag); errNum |= flag;

    m_ckSET  = clCreateKernel(m_program, "set",  &flag); errNum |= flag;
    m_ckCOPY = clCreateKernel(m_program, "copy", &flag); errNum |= flag;
    m_ckAXPY = clCreateKernel(m_program, "axpy", &flag); errNum |= flag;
    m_ckDOT  = clCreateKernel(m_program, "dotp", &flag); errNum |= flag;
    m_ckGEMV = clCreateKernel(m_program, "gemv", &flag); errNum |= flag;
    m_ckGBMV = clCreateKernel(m_program, "gbmv", &flag); errNum |= flag;
    m_ckSCL  = clCreateKernel(m_program, "scl",  &flag); errNum |= flag;
    m_ckSPAI = clCreateKernel(m_program, "spai", &flag); errNum |= flag;

    if(errNum!=CL_SUCCESS)
        throw std::runtime_error("CL_INVALID_KERNEL");

    cout << "Created kernels." << endl;
}


PressureBasedSegregatedSolver::~PressureBasedSegregatedSolver()
{
    clReleaseEvent(m_event);
    clReleaseKernel(m_ckBoundaryValues);
    clReleaseKernel(m_ckFaceValues);
    clReleaseKernel(m_ckComputeGradient1);
    clReleaseKernel(m_ckComputeGradient2);
    clReleaseKernel(m_ckComputeLimiter1);
    clReleaseKernel(m_ckComputeLimiter2);
    clReleaseKernel(m_ckAssembleMomMat1);
    clReleaseKernel(m_ckAssembleMomMat2);
    clReleaseKernel(m_ckAssembleMomMat3);
    clReleaseKernel(m_ckComp2ndOrderFlux);
    clReleaseKernel(m_ckComputeMomSrc1);
    clReleaseKernel(m_ckAssemblePCmat1);
    clReleaseKernel(m_ckAssemblePCmat2);
    clReleaseKernel(m_ckAssemblePCmat3);
    clReleaseKernel(m_ckComputeMassFlows);
    clReleaseKernel(m_ckCorrectMassFlows);
    clReleaseKernel(m_ckBoundaryValuesPC);
    clReleaseKernel(m_ckCorrectVelocity);
    clReleaseKernel(m_ckSET);
    clReleaseKernel(m_ckCOPY);
    clReleaseKernel(m_ckAXPY);
    clReleaseKernel(m_ckDOT);
    clReleaseKernel(m_ckGEMV);
    clReleaseKernel(m_ckGBMV);
    clReleaseKernel(m_ckSCL);
    clReleaseKernel(m_ckSPAI);
    clReleaseProgram(m_program);
    clReleaseCommandQueue(m_cmdQueue);
    clReleaseContext(m_context);
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
    for(size_t j=0; j<m_mesh->faces.groups.size(); j++)
    {
        float_t velMagni;

        switch(m_mesh->faces.groups[j].first)
        {
          case 2: //Inlet

            switch(flowParams.m_inletConditions.variable)
            {
              case flowParams.VELOCITY:

                switch(flowParams.m_inletConditions.direction)
                {
                  case flowParams.NORMAL:

                    for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                    {
                        int_t face = m_mesh->faces.groups[j].second[k];
                        float_t tmp = flowParams.m_inletConditions.scalar/
                                      m_mesh->faces.area.row(face).norm();
                        m_boundaries.type(face) = 1;
                        m_boundaries.conditions.block(face,0,1,3) = -tmp*m_mesh->faces.area.row(face);
                    }
                    break;

                  case flowParams.COMPONENTS:

                    if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                    {
                        for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                        {
                            int_t face = m_mesh->faces.groups[j].second[k];
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = flowParams.m_inletConditions.components[0];
                            m_boundaries.conditions(face,1) = flowParams.m_inletConditions.components[1];
                            m_boundaries.conditions(face,2) = flowParams.m_inletConditions.components[2];
                        }
                    }else // CYLINDRICAL
                    {
                        for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                        {
                            int_t face = m_mesh->faces.groups[j].second[k];
                            float_t l = m_mesh->faces.centroid.block(face,0,1,2).norm();
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = 1.0/l*(
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[0]-
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,1) = 1.0/l*(
                                m_mesh->faces.centroid(face,1)*flowParams.m_inletConditions.components[0]+
                                m_mesh->faces.centroid(face,0)*flowParams.m_inletConditions.components[2]);
                            m_boundaries.conditions(face,2) = flowParams.m_inletConditions.components[1];
                        }
                    }
                    break;

                  case flowParams.DIRECTION:

                    velMagni = flowParams.m_inletConditions.scalar/sqrt(
                               pow(flowParams.m_inletConditions.components[0],2.0)+
                               pow(flowParams.m_inletConditions.components[1],2.0)+
                               pow(flowParams.m_inletConditions.components[2],2.0));

                    if(flowParams.m_inletConditions.coordinate == flowParams.CARTESIAN)
                    {
                        for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                        {
                            int_t face = m_mesh->faces.groups[j].second[k];
                            m_boundaries.type(face) = 1;
                            m_boundaries.conditions(face,0) = velMagni*flowParams.m_inletConditions.components[0];
                            m_boundaries.conditions(face,1) = velMagni*flowParams.m_inletConditions.components[1];
                            m_boundaries.conditions(face,2) = velMagni*flowParams.m_inletConditions.components[2];
                        }
                    }else // CYLINDRICAL
                    {
                        for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                        {
                            int_t face = m_mesh->faces.groups[j].second[k];
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

              default: // MASSFLOW, TOTALPRESSURE
                  throw std::runtime_error("Unsupported type of inlet.");
            }
            break;
          case 3: //Outlet
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
            {
                int face = m_mesh->faces.groups[j].second[k];
                m_boundaries.type(face) = 3;
                m_boundaries.conditions(face,3) = flowParams.m_outletConditions.pressure;
            }
            break;
          case 6: //Blade
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                m_boundaries.type(m_mesh->faces.groups[j].second[k]) = 5;
            break;
          case 0: //Bottom
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                m_boundaries.type(m_mesh->faces.groups[j].second[k]) = 5;
            break;
          case 1: //Top
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                m_boundaries.type(m_mesh->faces.groups[j].second[k]) = 5;
            break;
          case 4: //Periodic 1
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                m_boundaries.type(m_mesh->faces.groups[j].second[k]) = 5;
            break;
          case 5: //Periodic 2
            for(size_t k=0; k<m_mesh->faces.groups[j].second.size(); k++)
                m_boundaries.type(m_mesh->faces.groups[j].second[k]) = 5;
            break;
        }
    }
    m_bndrsSet = true;
    return 0;
}


int PressureBasedSegregatedSolver::run()
{
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

    m_buildAdjacency();
    m_allocate();
    if(!m_isInit) m_defaultInit();

    m_res.setOnes();
    m_resNorm.setOnes();

    double ttot = -omp_get_wtime();
    double tlin1 = 0.0, tlin2 = 0.0;

    // outer iterations
    for(int iter=1; iter<=m_control.maxIt; iter++)
    {
        // relaxation factors
        m_relaxV = 0.01*m_control.cflNumber;

        if(iter < m_control.minIt/2) {
            m_relaxP = m_control.relax0;
        } else if(iter < m_control.minIt) {
            // ramp relaxation factors
            float_t blend = 2.0*float_t(iter)/float_t(m_control.minIt)-1.0;
            m_relaxP = blend*m_control.relax1+(1.0-blend)*m_control.relax0;
        } else {
            m_relaxP = m_control.relax1;
        }

        // ### Momentum ###
        m_boundaryValues();
        m_assembleMomMat();
        m_computeGradient(&m_field.cl_pVar);

        #define SOLVEMOMENTUM(var) \
          m_computeGradient(&var);\
          m_computeLimiter(&var);\
          m_compute2ndOrderFluxes(&var);\
          m_computeMomSource(&var);\
          tlin1 -= omp_get_wtime();\
          m_solveMomentum(&var);\
          tlin1 += omp_get_wtime()
        SOLVEMOMENTUM(m_field.cl_uVar);
        SOLVEMOMENTUM(m_field.cl_vVar);
        SOLVEMOMENTUM(m_field.cl_wVar);
        #undef SOLVEMOMENTUM

        // ### Pressure Correction ###
        m_assemblePCmat();
        m_boundaryValues();
        m_computeMassFlows();
        tlin2 -= omp_get_wtime();
        m_solvePressureCorrection();
        tlin2 += omp_get_wtime();
        m_applyCorrections();

        // monitor convergence
        if(iter<3)
          for(int i=0; i<4; ++i)
            if((m_resNorm(i)==1.0 && m_res(i)>1e-9) || m_res(i)>5.0*m_resNorm(i))
              m_resNorm(i) = m_res(i);

        cout << iter << " - " << m_res/m_resNorm << std::endl;

        if((m_res/m_resNorm < m_control.tol).all() || !m_res.isFinite().all()) break;
    }

    ttot += omp_get_wtime();

    cout << ttot-tlin1-tlin2 << "  " << tlin1 << "  " << tlin2 << "  " << ttot << std::endl;

    // retrieve solution
    // variables are interpolated to face centroids on the device and copied 1 by 1

    size_t sz = sizeof(float_t)*m_geometry.cl_nFaces;
    float_t* dst = NULL;
    cl_int errNum;

    m_boundaryValues();

    #define GETSOL(var) \
      m_faceValues(&var);\
      errNum = clEnqueueReadBuffer(m_cmdQueue, *m_gradLimit.cl_phiF, CL_TRUE, 0, sz, dst, 0, NULL, NULL);\
      if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during getSolution")
    dst = m_solution.data();
    GETSOL(m_field.cl_uVar);

    dst += m_geometry.cl_nFaces;
    GETSOL(m_field.cl_vVar);

    dst += m_geometry.cl_nFaces;
    GETSOL(m_field.cl_wVar);

    dst += m_geometry.cl_nFaces;
    GETSOL(m_field.cl_pVar);
    #undef GETSOL

    m_finished = m_res.isFinite().all() && (m_res/m_resNorm < 1e5).all();
    if(!m_finished) cout << "The flow solver diverged" << std::endl;

    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "|                                                                                         |\n";
    cout << "|                                     SOLVER FINISHED                                     |\n";
    cout << "|                                                                                         |\n";
    cout << "-------------------------------------------------------------------------------------------\n";
    cout << "\n\n";

    return m_finished? 0:3;
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
            solution_p->row(vidx) += w*m_solution.row(fidx);
        }
    }

    for(int i=0; i<Nv; ++i) solution_p->row(i)/=sumOfWeights(i);

    return 0;
}


void PressureBasedSegregatedSolver::m_buildAdjacency()
{
    int Nf = m_mesh->faces.number,
        Nc = m_mesh->cells.number,
        Nb = m_boundaries.number;

    std::vector<std::vector<int> > faceIdx(Nc), cellIdx(Nc);

    m_adjacency.faceNum.setZero(Nc);
    m_adjacency.cellNum.setZero(Nc);

    // connectivity in contiguous storage
    m_adjacency.connect.resize(Nf,2);

    for(int i=Nb; i<Nf; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first,
            C1 = m_mesh->faces.connectivity[i].second;

        faceIdx[C0].push_back(i);
        cellIdx[C0].push_back(C1);
        m_adjacency.faceNum(C0) += 1;
        m_adjacency.cellNum(C0) += 1;

        faceIdx[C1].push_back(i);
        cellIdx[C1].push_back(C0);
        m_adjacency.faceNum(C1) += 1;
        m_adjacency.cellNum(C1) += 1;

        m_adjacency.connect(i,0) = C0;
        m_adjacency.connect(i,1) = C1;
    }

    for(int i=0; i<Nb; ++i)
    {
        int C0 = m_mesh->faces.connectivity[i].first;

        faceIdx[C0].push_back(i);
        m_adjacency.faceNum(C0) += 1;

        m_adjacency.connect(i,0) = C0;
        m_adjacency.connect(i,1) = -1;
    }

    int maxFaces = m_adjacency.faceNum.maxCoeff();

    // put face and cell indexes in contiguous storage
    m_adjacency.maxNeighbours = int_t(maxFaces);
    m_adjacency.faceIdx.resize(Nc,maxFaces);
    m_adjacency.cellIdx.resize(Nc,maxFaces);

    // determine relative direction (in/out, -1/1) of each cell's faces
    m_adjacency.faceDir.resize(Nc,maxFaces);

    for(int i=0; i<Nc; ++i)
    {
        // copy
        for(int j=0; j<m_adjacency.faceNum(i); ++j)
        {
            int face = faceIdx[i][j];
            int C0 = m_adjacency.connect(face,0);

            m_adjacency.faceIdx(i,j) = face;
            m_adjacency.faceDir(i,j) = 2*(i==C0)-1;
        }

        for(int j=0; j<m_adjacency.cellNum(i); ++j)
            m_adjacency.cellIdx(i,j) = cellIdx[i][j];

        // pad with valid indexes
        for(int j=m_adjacency.faceNum(i); j<maxFaces; ++j)
        {
            m_adjacency.faceIdx(i,j) = m_adjacency.faceIdx(i,0);
            m_adjacency.faceDir(i,j) = 0;
        }

        for(int j=m_adjacency.cellNum(i); j<maxFaces; ++j)
            m_adjacency.cellIdx(i,j) = i;
    }
}


void PressureBasedSegregatedSolver::m_allocate()
{
    int Nc = m_mesh->cells.number,
        Nf = m_mesh->faces.number,
        Nb = m_boundaries.number;

    m_geometry.cl_nCells = Nc;
    m_geometry.cl_nFaces = Nf;
    m_boundaries.cl_number = Nb;
    m_adjacency.cl_width = m_adjacency.maxNeighbours;

    #define ROUNDUP(n,d) ((n+d-1)/d)*d
    m_glbSizeCells = ROUNDUP(Nc, m_localSize);
    m_glbSizeFaces = ROUNDUP(Nf, m_localSize);
    m_glbSizeBndrs = ROUNDUP(Nb, m_localSize);
    #undef ROUNDUP

    cl_int errNum;
    size_t sz;

    #define CHECKERROR \
      if(errNum!=CL_SUCCESS) throw std::runtime_error("CL_INVALID_MEM_OBJECT")

    #define ALLOCATE(DST,SZ) \
      DST = clCreateBuffer(m_context, CL_MEM_READ_WRITE, SZ, NULL, &errNum);\
      CHECKERROR

    #define INITIALIZE(DST,SRC,SZ) \
      DST = clCreateBuffer(m_context, CL_MEM_READ_ONLY, SZ, NULL, &errNum);\
      CHECKERROR;\
      errNum = clEnqueueWriteBuffer(m_cmdQueue,DST,CL_FALSE,0,SZ,(void*)SRC,0,NULL,NULL);\
      CHECKERROR

    // buffers for geometric properties
    sz = Nf*sizeof(float_t);
    INITIALIZE(m_geometry.cl_wf,   m_mesh->faces.wf.data(),    sz);
    INITIALIZE(m_geometry.cl_a0,   m_mesh->faces.alfa0.data(), sz);
    INITIALIZE(m_geometry.cl_a1,   m_mesh->faces.alfa1.data(), sz);
    sz = 3*sz;
    INITIALIZE(m_geometry.cl_area, m_mesh->faces.area.data(),  sz);
    INITIALIZE(m_geometry.cl_r0,   m_mesh->faces.r0.data(),    sz);
    INITIALIZE(m_geometry.cl_r1,   m_mesh->faces.r1.data(),    sz);
    INITIALIZE(m_geometry.cl_d0,   m_mesh->faces.d0.data(),    sz);
    INITIALIZE(m_geometry.cl_d1,   m_mesh->faces.d1.data(),    sz);

    sz = Nc*sizeof(float_t);
    INITIALIZE(m_geometry.cl_volume, m_mesh->cells.volume.data(),sz);

    // buffers for adjacency and connectivity
    sz = Nc*sizeof(int_t);
    INITIALIZE(m_adjacency.cl_cellNum, m_adjacency.cellNum.data(), sz);

    sz = sz*m_adjacency.maxNeighbours;
    INITIALIZE(m_adjacency.cl_cellIdx, m_adjacency.cellIdx.data(), sz);
    INITIALIZE(m_adjacency.cl_faceIdx, m_adjacency.faceIdx.data(), sz);

    sz = (sz/sizeof(int_t))*sizeof(float_t);
    INITIALIZE(m_adjacency.cl_faceDir, m_adjacency.faceDir.data(), sz);

    sz = 2*Nf*sizeof(int_t);
    INITIALIZE(m_adjacency.cl_connect, m_adjacency.connect.data(), sz);

    // boundary buffers
    sz = Nb*sizeof(int_t);
    INITIALIZE(m_boundaries.cl_type, m_boundaries.type.data(), sz);

    sz = 5*Nb*sizeof(float_t); // u,v,w,p + p'
    ALLOCATE(m_boundaries.cl_values, sz);
    sz = sz*4/5; // p' is not copied
    errNum = clEnqueueWriteBuffer(m_cmdQueue, m_boundaries.cl_values, CL_FALSE,
             0, sz, (void*)m_boundaries.conditions.data(), 0, NULL, NULL);
    CHECKERROR;

    // memory pool
    sz = Nf*sizeof(float_t);
    ALLOCATE(m_memoryPool.cl_faceBuf1, sz);
    ALLOCATE(m_memoryPool.cl_faceBuf2, sz);
    ALLOCATE(m_memoryPool.cl_faceBuf3, sz);

    sz = Nc*sizeof(float_t);
    ALLOCATE(m_memoryPool.cl_cellBuf1, sz);

    sz = sz*m_adjacency.maxNeighbours;
    ALLOCATE(m_memoryPool.cl_largeBuf, sz);

    // buffers for field variables, gradients, and limiters
    sz = Nc*sizeof(float_t);
    ALLOCATE(m_field.cl_u, sz);
    ALLOCATE(m_field.cl_v, sz);
    ALLOCATE(m_field.cl_w, sz);
    ALLOCATE(m_field.cl_p, sz);

    sz = Nf*sizeof(float_t);
    ALLOCATE(m_field.cl_mdot, sz);

    sz = 3*Nc*sizeof(float_t);
    ALLOCATE(m_gradLimit.cl_gradP, sz);

    m_gradLimit.cl_gradU = &m_memoryPool.cl_largeBuf;
    m_gradLimit.cl_limit = &m_memoryPool.cl_cellBuf1;
    m_gradLimit.cl_phiF  = &m_memoryPool.cl_faceBuf1;
    m_gradLimit.cl_flux  = &m_memoryPool.cl_faceBuf2;
    m_gradLimit.cl_delta = &m_memoryPool.cl_faceBuf1;
    m_gradLimit.cl_proj0 = &m_memoryPool.cl_faceBuf2;
    m_gradLimit.cl_proj1 = &m_memoryPool.cl_faceBuf3;

    // linear system
    sz = Nc*sizeof(float_t);
    ALLOCATE(m_linearSys.cl_diagonal, sz);

    sz = sz*m_adjacency.maxNeighbours;
    ALLOCATE(m_linearSys.cl_offDiag, sz);

    sz = Nf*sizeof(float_t);
    ALLOCATE(m_linearSys.cl_diffCoeff, sz);

    sz = m_glbSizeCells/m_localSize;
    m_linearSys.partDot.resize(sz);

    m_linearSys.sizePartDot = sz*sizeof(float_t);
    ALLOCATE(m_linearSys.cl_partDot, m_linearSys.sizePartDot);

    m_linearSys.cl_diagonalSPAI = &m_memoryPool.cl_faceBuf1;
    m_linearSys.cl_offDiagSPAI = &m_memoryPool.cl_largeBuf;

    m_linearSys.cl_faceType = &m_memoryPool.cl_faceBuf1;
    m_linearSys.cl_corrFlux = &m_memoryPool.cl_faceBuf2;
    m_linearSys.cl_diagCopy = &m_memoryPool.cl_cellBuf1;
    m_linearSys.cl_solution = &m_memoryPool.cl_faceBuf2;
    m_linearSys.cl_source   = &m_memoryPool.cl_faceBuf3;

    // linear solver
    sz = Nc*sizeof(float_t);
    ALLOCATE(m_linearSol.cl_v, sz);
    ALLOCATE(m_linearSol.cl_p, sz);

    m_linearSol.cl_y  = &m_memoryPool.cl_largeBuf;
    m_linearSol.cl_z  = &m_memoryPool.cl_cellBuf1;
    m_linearSol.cl_r  = m_linearSys.cl_source;
    m_linearSol.cl_r0 = &m_memoryPool.cl_faceBuf1;

    // solution (interpolated at face centroids)
    m_solution.resize(Nf,4);

    #undef INITIALIZE
    #undef ALLOCATE
    #undef CHECKERROR
}


void PressureBasedSegregatedSolver::m_defaultInit()
{
    // set velocity and face mass flows to zero, and pressure to maximum value
    cl_int errNum = CL_SUCCESS;

    m_const = -1.0e9f;
    for(int j=0; j<m_boundaries.number; j++)
      if(m_boundaries.type(j)==3)
        m_const = std::max(m_const,m_boundaries.conditions(j,3));

    errNum |= m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_zero,  &m_field.cl_u);
    errNum |= m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_zero,  &m_field.cl_v);
    errNum |= m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_zero,  &m_field.cl_w);
    errNum |= m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_const, &m_field.cl_p);
    errNum |= m_blas_SET(m_glbSizeFaces, m_geometry.cl_nFaces, m_zero,  &m_field.cl_mdot);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during defaultInit");

    m_isInit = true;
}


void PressureBasedSegregatedSolver::m_boundaryValues()
{
    cl_int errNum = CL_SUCCESS;

    errNum |= clSetKernelArg(m_ckBoundaryValues, 0, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 1, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 3, sizeof(cl_mem),  (void*)&m_field.cl_u);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 4, sizeof(cl_mem),  (void*)&m_field.cl_v);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 5, sizeof(cl_mem),  (void*)&m_field.cl_w);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 6, sizeof(cl_mem),  (void*)&m_field.cl_p);
    errNum |= clSetKernelArg(m_ckBoundaryValues, 7, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during boundaryValues");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckBoundaryValues, 1, NULL,
                                    &m_glbSizeBndrs, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during boundaryValues");
}


void PressureBasedSegregatedSolver::m_faceValues(const cl_uint* var)
{
    cl_int errNum = CL_SUCCESS;

    cl_mem* phiC = NULL;
    if(*var==m_field.cl_uVar) phiC = &m_field.cl_u;
    if(*var==m_field.cl_vVar) phiC = &m_field.cl_v;
    if(*var==m_field.cl_wVar) phiC = &m_field.cl_w;
    if(*var==m_field.cl_pVar) phiC = &m_field.cl_p;
    if(*var==m_field.cl_pcVar) phiC = m_linearSys.cl_solution;

    errNum |= clSetKernelArg(m_ckFaceValues, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckFaceValues, 1, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckFaceValues, 2, sizeof(cl_uint), (void*)var);
    errNum |= clSetKernelArg(m_ckFaceValues, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckFaceValues, 4, sizeof(cl_mem),  (void*)&m_geometry.cl_wf);
    errNum |= clSetKernelArg(m_ckFaceValues, 5, sizeof(cl_mem),  (void*)phiC);
    errNum |= clSetKernelArg(m_ckFaceValues, 6, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);
    errNum |= clSetKernelArg(m_ckFaceValues, 7, sizeof(cl_mem),  (void*)m_gradLimit.cl_phiF);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during faceValues");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckFaceValues, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during faceValues");
}


void PressureBasedSegregatedSolver::m_computeGradient(const cl_uint* var)
{
    cl_int errNum = CL_SUCCESS;

    // velocity gradients are stored in "gradU" as they are temporary, only used
    // to compute the corresponding source term, the pressure gradient "gradP" is
    // persistent as is is common to all velocities and later used for dissipation
    // and corrections (gradient of p')
    cl_mem* grad_ptr = NULL;
    if(*var==m_field.cl_pVar || *var==m_field.cl_pcVar)
        grad_ptr = &m_gradLimit.cl_gradP;
    else
        grad_ptr = m_gradLimit.cl_gradU;

    // interpolate face values for target variable

    m_faceValues(var);

    // compute flux and sum for each direction

    for(cl_uint j=0; j<3; ++j)
    {
    cl_uint* dim = NULL;
    switch(j) {
    case 0: dim = &m_geometry.cl_xDim; break;
    case 1: dim = &m_geometry.cl_yDim; break;
    case 2: dim = &m_geometry.cl_zDim; break;
    }
    // flux
    errNum |= clSetKernelArg(m_ckComputeGradient1, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckComputeGradient1, 1, sizeof(cl_uint), (void*)dim);
    errNum |= clSetKernelArg(m_ckComputeGradient1, 2, sizeof(cl_mem),  (void*)&m_geometry.cl_area);
    errNum |= clSetKernelArg(m_ckComputeGradient1, 3, sizeof(cl_mem),  (void*)m_gradLimit.cl_phiF);
    errNum |= clSetKernelArg(m_ckComputeGradient1, 4, sizeof(cl_mem),  (void*)m_gradLimit.cl_flux);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeGradient");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeGradient1, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeGradient");

    // summation
    errNum |= clSetKernelArg(m_ckComputeGradient2, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 1, sizeof(cl_uint), (void*)&m_adjacency.cl_width);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 2, sizeof(cl_uint), (void*)dim);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceIdx);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 4, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceDir);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 5, sizeof(cl_mem),  (void*)m_gradLimit.cl_flux);
    errNum |= clSetKernelArg(m_ckComputeGradient2, 6, sizeof(cl_mem),  (void*)grad_ptr);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeGradient");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeGradient2, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeGradient");
    }

    // divide by the volume (except the pressure correction gradient)

    if(*var==m_field.cl_pcVar) return;

    errNum = m_blas_SCL(m_glbSizeCells, m_geometry.cl_nCells,
             m_geometry.cl_nDim, &m_geometry.cl_volume, grad_ptr);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeGradient");
}


void PressureBasedSegregatedSolver::m_computeLimiter(const cl_uint* var)
{
    cl_int errNum = CL_SUCCESS;

    // compute deltas and projections for each face

    cl_mem* phiC = NULL;
    if(*var==m_field.cl_uVar) phiC = &m_field.cl_u;
    if(*var==m_field.cl_vVar) phiC = &m_field.cl_v;
    if(*var==m_field.cl_wVar) phiC = &m_field.cl_w;

    if(*var==m_field.cl_pVar || *var==m_field.cl_pcVar)
        if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeLimiter");

    errNum |= clSetKernelArg(m_ckComputeLimiter1, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 1, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 2, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 3, sizeof(cl_uint), (void*)var);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 4, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 5, sizeof(cl_mem),  (void*)&m_geometry.cl_r0);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 6, sizeof(cl_mem),  (void*)&m_geometry.cl_r1);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 7, sizeof(cl_mem),  (void*)phiC);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 8, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);
    errNum |= clSetKernelArg(m_ckComputeLimiter1, 9, sizeof(cl_mem),  (void*)m_gradLimit.cl_gradU);
    errNum |= clSetKernelArg(m_ckComputeLimiter1,10, sizeof(cl_mem),  (void*)m_gradLimit.cl_delta);
    errNum |= clSetKernelArg(m_ckComputeLimiter1,11, sizeof(cl_mem),  (void*)m_gradLimit.cl_proj0);
    errNum |= clSetKernelArg(m_ckComputeLimiter1,12, sizeof(cl_mem),  (void*)m_gradLimit.cl_proj1);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeLimiter");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeLimiter1, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeLimiter");

    // determine min/max delta/projection for each cell and apply formula

    errNum |= clSetKernelArg(m_ckComputeLimiter2, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 1, sizeof(cl_uint), (void*)&m_adjacency.cl_width);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceIdx);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceDir);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 4, sizeof(cl_mem),  (void*)m_gradLimit.cl_delta);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 5, sizeof(cl_mem),  (void*)m_gradLimit.cl_proj0);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 6, sizeof(cl_mem),  (void*)m_gradLimit.cl_proj1);
    errNum |= clSetKernelArg(m_ckComputeLimiter2, 7, sizeof(cl_mem),  (void*)m_gradLimit.cl_limit);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeLimiter");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeLimiter2, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeLimiter");
}


void PressureBasedSegregatedSolver::m_assembleMomMat()
{
    cl_int errNum = CL_SUCCESS;

    // compute diffusion coefficient for each face

    errNum |= clSetKernelArg(m_ckAssembleMomMat1, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckAssembleMomMat1, 1, sizeof(float_t), (void*)&m_mu);
    errNum |= clSetKernelArg(m_ckAssembleMomMat1, 2, sizeof(cl_mem),  (void*)&m_geometry.cl_a0);
    errNum |= clSetKernelArg(m_ckAssembleMomMat1, 3, sizeof(cl_mem),  (void*)&m_geometry.cl_a1);
    errNum |= clSetKernelArg(m_ckAssembleMomMat1, 4, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssembleMomMat1, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");

    // handle Dirichlet boundaries

    errNum |= clSetKernelArg(m_ckAssembleMomMat2, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckAssembleMomMat2, 1, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckAssembleMomMat2, 2, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckAssembleMomMat2, 3, sizeof(cl_mem),  (void*)m_linearSys.cl_faceType);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssembleMomMat2, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");

    // set on and off diagonal coefficients

    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 1, sizeof(cl_uint), (void*)&m_adjacency.cl_width);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 2, sizeof(float_t), (void*)&m_relaxV);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceIdx);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 4, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceDir);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 5, sizeof(cl_mem),  (void*)m_linearSys.cl_faceType);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 6, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 7, sizeof(cl_mem),  (void*)&m_field.cl_mdot);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 8, sizeof(cl_mem),  (void*)&m_linearSys.cl_offDiag);
    errNum |= clSetKernelArg(m_ckAssembleMomMat3, 9, sizeof(cl_mem),  (void*)&m_linearSys.cl_diagonal);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssembleMomMat3, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assembleMomMat");
}


void PressureBasedSegregatedSolver::m_compute2ndOrderFluxes(const cl_uint* var)
{
    cl_int errNum = CL_SUCCESS;

    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 1, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 2, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 3, sizeof(cl_uint), (void*)var);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 4, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 5, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 6, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 7, sizeof(cl_mem),  (void*)&m_geometry.cl_r0);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 8, sizeof(cl_mem),  (void*)&m_geometry.cl_r1);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux, 9, sizeof(cl_mem),  (void*)&m_geometry.cl_d0);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,10, sizeof(cl_mem),  (void*)&m_geometry.cl_d1);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,11, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,12, sizeof(cl_mem),  (void*)&m_field.cl_mdot);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,13, sizeof(cl_mem),  (void*)m_gradLimit.cl_gradU);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,14, sizeof(cl_mem),  (void*)m_gradLimit.cl_limit);
    errNum |= clSetKernelArg(m_ckComp2ndOrderFlux,15, sizeof(cl_mem),  (void*)m_linearSys.cl_corrFlux);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during compute2ndOrderFluxes");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComp2ndOrderFlux, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during compute2ndOrderFluxes");
}


void PressureBasedSegregatedSolver::m_computeMomSource(const cl_uint* var)
{
    cl_int errNum = CL_SUCCESS;

    // initialize with relaxation term and pressure gradient

    cl_uint* dim = NULL;
    if(*var==m_field.cl_uVar) dim = &m_geometry.cl_xDim;
    if(*var==m_field.cl_vVar) dim = &m_geometry.cl_yDim;
    if(*var==m_field.cl_wVar) dim = &m_geometry.cl_zDim;

    cl_mem* phiC = NULL;
    if(*var==m_field.cl_uVar) phiC = &m_field.cl_u;
    if(*var==m_field.cl_vVar) phiC = &m_field.cl_v;
    if(*var==m_field.cl_wVar) phiC = &m_field.cl_w;

    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 1, sizeof(cl_uint), (void*)dim);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 2, sizeof(float_t), (void*)&m_relaxV);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 3, sizeof(cl_mem),  (void*)&m_linearSys.cl_diagonal);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 4, sizeof(cl_mem),  (void*)phiC);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 5, sizeof(cl_mem),  (void*)&m_gradLimit.cl_gradP);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 6, sizeof(cl_mem),  (void*)&m_geometry.cl_volume);
    errNum |= clSetKernelArg(m_ckComputeMomSrc1, 7, sizeof(cl_mem),  (void*)m_linearSys.cl_source);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeMomSrc1, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");

    // add sum of fluxes

    errNum = m_blas_GEMV(m_glbSizeCells, m_geometry.cl_nCells, m_adjacency.cl_width,
             m_plusOne, m_plusOne, &m_adjacency.cl_faceIdx, &m_adjacency.cl_faceDir,
             m_linearSys.cl_corrFlux, m_linearSys.cl_source);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");

    // convert to residual, diagonal and off-diagonal contributions are accounted separately

    errNum = m_blas_GBMV(m_glbSizeCells, m_geometry.cl_nCells, m_minusOne, m_plusOne,
                         &m_linearSys.cl_diagonal, phiC, m_linearSys.cl_source);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");

    errNum = m_blas_GEMV(m_glbSizeCells, m_geometry.cl_nCells, m_adjacency.cl_width,
                         m_minusOne, m_plusOne, &m_adjacency.cl_cellIdx,
                         &m_linearSys.cl_offDiag, phiC, m_linearSys.cl_source);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");

    // norm of residual

    errNum = m_blas_DOT(m_linearSys.cl_source, m_linearSys.cl_source, m_const);
    m_res(*var) = std::sqrt(m_const);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMomSource");
}


void PressureBasedSegregatedSolver::m_solveMomentum(const cl_uint* var)
{
    /* BiCGSTAB with diagonal preconditioning

    bNorm = norm(b) <- this is known at this stage (residual!)
    x = 0
    r = b <- since x=0, also work directly on b instead of copying
    r0 = r
    alpha = omega = rho = 1
    v = p = 0

    for...
        gamma = r0.r
        beta = (gamma/rho)*(alpha/omega)
        rho = gamma
        z = p
        z = z-w*v
        p = r
        p = p+beta*z
        y = D^-1*p
        v = A*y
        alpha = r0.v
        alpha = rho/alpha
        x = x+alpha*p
        r = r-alpha*v
        y = D^-1*r
        z = A*y
        gamma = z.z
        omega = z.r
        omega = omega/gamma
        x = x+omega*r
        r = r-omega*z

        rNorm = norm(r)
        if rNorm/bNorm < tol: break
    end */

    cl_int errNum = CL_SUCCESS;
    cl_uint nRHS = 1;

    #define CHECKERROR\
      if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during solveMomentum")

    #define SETZERO(VEC)\
      errNum = m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_zero, VEC);\
      CHECKERROR

    #define COPY(SRC,DST)\
      errNum = m_blas_COPY(m_glbSizeCells, m_geometry.cl_nCells, SRC, DST);\
      CHECKERROR

    #define DOT(U,V,RES)\
      errNum = m_blas_DOT(U,V,RES);\
      CHECKERROR

    #define AXPY(A,X,Y)\
      errNum = m_blas_AXPY(m_glbSizeCells, m_geometry.cl_nCells, A, X, Y);\
      CHECKERROR

    // y = A * (D^-1 * x)
    #define MULT(X,Y)\
      COPY(X,m_linearSol.cl_y);\
      errNum = m_blas_SCL(m_glbSizeCells, m_geometry.cl_nCells, nRHS,\
                          &m_linearSys.cl_diagonal, m_linearSol.cl_y);\
      CHECKERROR;\
      COPY(X,Y);\
      errNum = m_blas_GEMV(m_glbSizeCells, m_geometry.cl_nCells, m_adjacency.cl_width,\
               m_plusOne, m_plusOne, &m_adjacency.cl_cellIdx, &m_linearSys.cl_offDiag,\
               m_linearSol.cl_y, Y);\
      CHECKERROR

    // Solve system

    float_t alpha, beta, omega, rho, gamma, rNormSqr, bNorm = m_res(*var);

    if(bNorm < 10*std::numeric_limits<float_t>::min()) return;

    alpha = omega = rho = 1.0;

    SETZERO( m_linearSys.cl_solution);
    SETZERO(&m_linearSol.cl_v);
    SETZERO(&m_linearSol.cl_p);

    COPY(m_linearSol.cl_r, m_linearSol.cl_r0);

    for(int i=0; i<m_control.linSolMaxIt; ++i)
    {
        gamma = rho;
        DOT(m_linearSol.cl_r0, m_linearSol.cl_r, rho);

        beta = (rho/gamma)*(alpha/omega);

        gamma = -omega;
        COPY(&m_linearSol.cl_p, m_linearSol.cl_z);
        AXPY(gamma, &m_linearSol.cl_v, m_linearSol.cl_z);
        COPY(m_linearSol.cl_r, &m_linearSol.cl_p);
        AXPY(beta, m_linearSol.cl_z, &m_linearSol.cl_p);

        MULT(&m_linearSol.cl_p, &m_linearSol.cl_v);

        DOT(m_linearSol.cl_r0, &m_linearSol.cl_v, alpha);
        alpha = rho/alpha;
        gamma = -alpha;

        AXPY(alpha,  m_linearSol.cl_y, m_linearSys.cl_solution);
        AXPY(gamma, &m_linearSol.cl_v, m_linearSol.cl_r);

        MULT(m_linearSol.cl_r, m_linearSol.cl_z);

        DOT(m_linearSol.cl_z, m_linearSol.cl_z, gamma);
        DOT(m_linearSol.cl_z, m_linearSol.cl_r, omega);
        omega /= gamma;
        gamma = -omega;

        AXPY(omega, m_linearSol.cl_y, m_linearSys.cl_solution);
        AXPY(gamma, m_linearSol.cl_z, m_linearSol.cl_r);

        DOT(m_linearSol.cl_r, m_linearSol.cl_r, rNormSqr);
        if(std::sqrt(rNormSqr)/bNorm < m_control.linSolTol) break;
    }

    // Update velocity component
    if(*var==m_field.cl_uVar) {AXPY(m_plusOne, m_linearSys.cl_solution, &m_field.cl_u);}
    if(*var==m_field.cl_vVar) {AXPY(m_plusOne, m_linearSys.cl_solution, &m_field.cl_v);}
    if(*var==m_field.cl_wVar) {AXPY(m_plusOne, m_linearSys.cl_solution, &m_field.cl_w);}

    #undef SETZERO
    #undef COPY
    #undef DOT
    #undef AXPY
    #undef MULT
    #undef CHECKERROR
}


void PressureBasedSegregatedSolver::m_assemblePCmat()
{
    cl_int errNum = CL_SUCCESS;

    // compute diffusion coefficient for each face

    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 1, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 2, sizeof(float_t), (void*)&m_rho);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 4, sizeof(cl_mem),  (void*)&m_geometry.cl_a0);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 5, sizeof(cl_mem),  (void*)&m_geometry.cl_a1);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 6, sizeof(cl_mem),  (void*)&m_geometry.cl_volume);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 7, sizeof(cl_mem),  (void*)&m_linearSys.cl_diagonal);
    errNum |= clSetKernelArg(m_ckAssemblePCmat1, 8, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssemblePCmat1, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    // make a copy of the momentum diagonal as it is needed later for the corrections

    errNum = m_blas_COPY(m_glbSizeCells, m_geometry.cl_nCells,
             &m_linearSys.cl_diagonal, m_linearSys.cl_diagCopy);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    // handle Dirichlet boundaries

    errNum |= clSetKernelArg(m_ckAssemblePCmat2, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckAssemblePCmat2, 1, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckAssemblePCmat2, 2, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckAssemblePCmat2, 3, sizeof(cl_mem),  (void*)m_linearSys.cl_faceType);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssemblePCmat2, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    // set on and off diagonal coefficients

    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 1, sizeof(cl_uint), (void*)&m_adjacency.cl_width);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceIdx);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_faceDir);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 4, sizeof(cl_mem),  (void*)m_linearSys.cl_faceType);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 5, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 6, sizeof(cl_mem),  (void*)&m_linearSys.cl_offDiag);
    errNum |= clSetKernelArg(m_ckAssemblePCmat3, 7, sizeof(cl_mem),  (void*)&m_linearSys.cl_diagonal);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckAssemblePCmat3, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during assemblePCmat");
}


void PressureBasedSegregatedSolver::m_computeMassFlows()
{
    cl_int errNum = CL_SUCCESS;

    errNum |= clSetKernelArg(m_ckComputeMassFlows, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 1, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 2, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 3, sizeof(float_t), (void*)&m_rho);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 4, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 5, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 6, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 7, sizeof(cl_mem),  (void*)&m_geometry.cl_wf);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 8, sizeof(cl_mem),  (void*)&m_geometry.cl_r0);
    errNum |= clSetKernelArg(m_ckComputeMassFlows, 9, sizeof(cl_mem),  (void*)&m_geometry.cl_r1);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,10, sizeof(cl_mem),  (void*)&m_geometry.cl_area);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,11, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,12, sizeof(cl_mem),  (void*)&m_field.cl_u);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,13, sizeof(cl_mem),  (void*)&m_field.cl_v);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,14, sizeof(cl_mem),  (void*)&m_field.cl_w);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,15, sizeof(cl_mem),  (void*)&m_field.cl_p);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,16, sizeof(cl_mem),  (void*)&m_gradLimit.cl_gradP);
    errNum |= clSetKernelArg(m_ckComputeMassFlows,17, sizeof(cl_mem),  (void*)&m_field.cl_mdot);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMassFlows");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckComputeMassFlows, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMassFlows");

    // compute the mass imbalance (pressure correction source term)

    errNum = m_blas_GEMV(m_glbSizeCells, m_geometry.cl_nCells, m_adjacency.cl_width, m_plusOne,
                         m_zero, &m_adjacency.cl_faceIdx, &m_adjacency.cl_faceDir,
                         &m_field.cl_mdot, m_linearSys.cl_source);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMassFlows");

    // norm of imbalance

    errNum = m_blas_DOT(m_linearSys.cl_source, m_linearSys.cl_source, m_const);
    m_res(3) = std::sqrt(m_const);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during computeMassFlows");
}


void PressureBasedSegregatedSolver::m_solvePressureCorrection()
{
    /* Conjugate Gradient preconditioned with SPAI

    bNorm = norm(b) <- this is known at this stage (residual!)
    x = 0
    r = b <- since x=0, also work directly on b instead of copying
    v = M*r
    p = v
    gamma = r.v

    for...
        v = A*p
        alpha = p.v
        alpha = gamma/alpha
        x = x+alpha*p
        r = r-alpha*v

        rNorm = norm(r)
        if rNorm/bNorm < tol: break

        v = M*r
        beta = gamma
        gamma = r.v
        beta = gamma/beta
        z = z+beta*p
        p = z
    end */

    cl_int errNum = CL_SUCCESS;

    #define CHECKERROR\
      if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during solvePressureCorrection")

    #define SETZERO(VEC)\
      errNum = m_blas_SET(m_glbSizeCells, m_geometry.cl_nCells, m_zero, VEC);\
      CHECKERROR

    #define COPY(SRC,DST)\
      errNum = m_blas_COPY(m_glbSizeCells, m_geometry.cl_nCells, SRC, DST);\
      CHECKERROR

    #define DOT(U,V,RES)\
      errNum = m_blas_DOT(U,V,RES);\
      CHECKERROR

    #define AXPY(A,X,Y)\
      errNum = m_blas_AXPY(m_glbSizeCells, m_geometry.cl_nCells, A, X, Y);\
      CHECKERROR

    #define MULT(AII,AIJ,X,Y)\
      errNum = m_blas_GBMV(m_glbSizeCells, m_geometry.cl_nCells, m_plusOne, m_zero, AII, X, Y);\
      CHECKERROR;\
      errNum = m_blas_GEMV(m_glbSizeCells, m_geometry.cl_nCells, m_adjacency.cl_width,\
                           m_plusOne, m_plusOne, &m_adjacency.cl_cellIdx, AIJ, X, Y);\
      CHECKERROR

    // Compute preconditioner

    errNum |= clSetKernelArg(m_ckSPAI, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckSPAI, 1, sizeof(cl_uint), (void*)&m_adjacency.cl_width);
    errNum |= clSetKernelArg(m_ckSPAI, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_cellNum);
    errNum |= clSetKernelArg(m_ckSPAI, 3, sizeof(cl_mem),  (void*)&m_adjacency.cl_cellIdx);
    errNum |= clSetKernelArg(m_ckSPAI, 4, sizeof(cl_mem),  (void*)&m_linearSys.cl_diagonal);
    errNum |= clSetKernelArg(m_ckSPAI, 5, sizeof(cl_mem),  (void*)&m_linearSys.cl_offDiag);
    errNum |= clSetKernelArg(m_ckSPAI, 6, sizeof(cl_mem),  (void*)m_linearSys.cl_diagonalSPAI);
    errNum |= clSetKernelArg(m_ckSPAI, 7, sizeof(cl_mem),  (void*)m_linearSys.cl_offDiagSPAI);
    CHECKERROR;
    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckSPAI,
             1, NULL, &m_glbSizeCells, &m_localSize, 0, NULL, NULL);
    CHECKERROR;

    // Solve system

    float_t alpha, beta, gamma, omega, rNormSqr, bNorm = m_res(3);

    SETZERO(m_linearSys.cl_solution);

    MULT(m_linearSys.cl_diagonalSPAI, m_linearSys.cl_offDiagSPAI, m_linearSol.cl_r, &m_linearSol.cl_v);

    COPY(&m_linearSol.cl_v, &m_linearSol.cl_p);

    DOT(m_linearSol.cl_r, &m_linearSol.cl_v, gamma);

    for(int i=0; i<m_control.linSolMaxIt; ++i)
    {
        MULT(&m_linearSys.cl_diagonal, &m_linearSys.cl_offDiag, &m_linearSol.cl_p, &m_linearSol.cl_v);

        DOT(&m_linearSol.cl_p, &m_linearSol.cl_v, alpha);
        alpha = gamma/alpha;
        omega = -alpha;

        AXPY(alpha, &m_linearSol.cl_p, m_linearSys.cl_solution);
        AXPY(omega, &m_linearSol.cl_v, m_linearSol.cl_r);

        DOT(m_linearSol.cl_r, m_linearSol.cl_r, rNormSqr);

        if(std::sqrt(rNormSqr)/bNorm < m_control.linSolTol) break;

        MULT(m_linearSys.cl_diagonalSPAI, m_linearSys.cl_offDiagSPAI, m_linearSol.cl_r, &m_linearSol.cl_v);

        beta = gamma;
        DOT(m_linearSol.cl_r, &m_linearSol.cl_v, gamma);
        beta = gamma/beta;

        AXPY(beta, &m_linearSol.cl_p, &m_linearSol.cl_v);
        COPY(&m_linearSol.cl_v, &m_linearSol.cl_p);
    }

    AXPY(m_relaxP, m_linearSys.cl_solution, &m_field.cl_p);

    #undef SETZERO
    #undef COPY
    #undef DOT
    #undef AXPY
    #undef MULT
    #undef CHECKERROR
}


void PressureBasedSegregatedSolver::m_applyCorrections()
{
    cl_int errNum = CL_SUCCESS;

    // ### Face mass flows ###

    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nFaces);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 1, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 3, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 4, sizeof(cl_mem),  (void*)&m_linearSys.cl_diffCoeff);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 5, sizeof(cl_mem),  (void*)m_linearSys.cl_solution);
    errNum |= clSetKernelArg(m_ckCorrectMassFlows, 6, sizeof(cl_mem),  (void*)&m_field.cl_mdot);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckCorrectMassFlows, 1, NULL,
                                    &m_glbSizeFaces, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");

    // ### Cell velocities ###

    // 1 - set boundary values of PC
    errNum |= clSetKernelArg(m_ckBoundaryValuesPC, 0, sizeof(cl_uint), (void*)&m_boundaries.cl_number);
    errNum |= clSetKernelArg(m_ckBoundaryValuesPC, 1, sizeof(cl_mem),  (void*)&m_boundaries.cl_type);
    errNum |= clSetKernelArg(m_ckBoundaryValuesPC, 2, sizeof(cl_mem),  (void*)&m_adjacency.cl_connect);
    errNum |= clSetKernelArg(m_ckBoundaryValuesPC, 3, sizeof(cl_mem),  (void*)m_linearSys.cl_solution);
    errNum |= clSetKernelArg(m_ckBoundaryValuesPC, 4, sizeof(cl_mem),  (void*)&m_boundaries.cl_values);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckBoundaryValuesPC, 1, NULL,
                                    &m_glbSizeBndrs, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");

    // 2 - compute PC gradient
    m_computeGradient(&m_field.cl_pcVar);

    // 3 - correct
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 0, sizeof(cl_uint), (void*)&m_geometry.cl_nCells);
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 1, sizeof(cl_mem),  (void*)m_linearSys.cl_diagCopy);
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 2, sizeof(cl_mem),  (void*)&m_gradLimit.cl_gradP);
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 3, sizeof(cl_mem),  (void*)&m_field.cl_u);
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 4, sizeof(cl_mem),  (void*)&m_field.cl_v);
    errNum |= clSetKernelArg(m_ckCorrectVelocity, 5, sizeof(cl_mem),  (void*)&m_field.cl_w);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");

    errNum = clEnqueueNDRangeKernel(m_cmdQueue, m_ckCorrectVelocity, 1, NULL,
                                    &m_glbSizeCells, &m_localSize, 0, NULL, NULL);

    if(errNum!=CL_SUCCESS) throw std::runtime_error("Error during applyCorrections");
}
}
