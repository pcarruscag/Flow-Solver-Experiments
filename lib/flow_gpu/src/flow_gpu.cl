//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

/*
 * Set the face values for boundary faces.
 * Only non-Dirichlet boundaries are set, others were done
 * when "bndValues" was initialized.
 * First order treatment of inlets and outlets, walls
 * are 2nd order if the corresponding cell is not skewed.
 * This kernel has branching, but is small and light.
 */
#define INLET  1
#define OUTLET 3
#define WALL   5

__kernel void boundaryValues(uint Nb,
                             const __global int* bndType,
                             const __global int* cellIdx,
                             const __global float* u,
                             const __global float* v,
                             const __global float* w,
                             const __global float* p,
                             __global float* bndValues)
{
    uint face = get_global_id(0);

    if(face>=Nb) return;

    int type = bndType[face];
    int cell = cellIdx[face];

    switch(type)
    {
      case INLET:
      case WALL:
        bndValues[face+3*Nb] = p[cell];
        break;

      case OUTLET:
        bndValues[face     ] = u[cell];
        bndValues[face + Nb] = v[cell];
        bndValues[face+2*Nb] = w[cell];
        break;
    }
}


/*
 * Linear interpolation of cell values onto faces
 */
__kernel void faceValues(uint Nf,
                         uint Nb,
                         uint var,
                         const __global int* connect,
                         const __global float* weight,
                         const __global float* phiC,
                         const __global float* phiB,
                         __global float* phiF)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0 = connect[face];
    int C1 = connect[face+Nf];
    float wf = weight[face];

    if(face<Nb)
        phiF[face] = phiB[face+var*Nb];
    else
        phiF[face] = wf*phiC[C0]+(1.0f-wf)*phiC[C1];
}


/*
 * First step to compute the gradient of a variable along a direction.
 * Compute the flux: "face value" times "area component".
 */
__kernel void computeGradient1(uint Nf,
                               uint dim,
                               const __global float* area,
                               const __global float* phiF,
                               __global float* flux)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    flux[face] = phiF[face]*area[face+dim*Nf];
}


/*
 * Second step of computing gradients, sum fluxes for each cell.
 */
__kernel void computeGradient2(uint Nc,
                               uint width,
                               uint dim,
                               const __global int* faceIdx,
                               const __global float* faceDir,
                               const __global float* flux,
                               __global float* gradient)
{
    uint cell = get_global_id(0);

    if(cell>=Nc) return;

    float sum = 0.0f;
    int face;
    uint idx;
    for(uint j=0; j<width; ++j)
    {
        idx = cell+j*Nc;
        face = faceIdx[idx];
        sum += faceDir[idx]*flux[face];
    }

    gradient[cell+dim*Nc] = sum;
}


/*
 * First step of computing a limiter.
 * Compute the face "deltas" and projections.
 */
__kernel void computeLimiter1(uint Nc,
                              uint Nf,
                              uint Nb,
                              uint var,
                              const __global int* connect,
                              const __global float* r0,
                              const __global float* r1,
                              const __global float* phiC,
                              const __global float* phiB,
                              const __global float* gradient,
                              __global float* delta,
                              __global float* proj0,
                              __global float* proj1)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0 = connect[face];
    int C1 = connect[face+Nf];

    float d = 0.0f, p0 = 0.0f, p1 = 0.0f;

    if(face<Nb)
    {
        d = phiB[face+var*Nb]-phiC[C0];

        for(int i=0; i<3; ++i) {
            p0 += r0[face+i*Nf]*gradient[C0+i*Nc];
        }
    }
    else
    {
        d = phiC[C1]-phiC[C0];

        for(int i=0; i<3; ++i) {
            p0 += r0[face+i*Nf]*gradient[C0+i*Nc];
            p1 += r1[face+i*Nf]*gradient[C1+i*Nc];
        }
    }

    delta[face] = d;
    proj0[face] = p0;
    proj1[face] = p1;
}


/*
 * Second step of computing a limiter.
 * Find max/min delta/proj for each cell and apply limiter formula.
 */
__kernel void computeLimiter2(uint Nc,
                              uint width,
                              const __global int* faceIdx,
                              const __global float* faceDir,
                              const __global float* delta,
                              const __global float* proj0,
                              const __global float* proj1,
                              __global float* limiter)
{
    uint cell = get_global_id(0);

    if(cell>=Nc) return;

    float dh = 0.0f,  dl = 0.0f,  d;
    float ph = 1e-9f, pl =-1e-9f, p;

    int idx;
    float dir;

    for(uint j=0; j<width; ++j)
    {
        idx = faceIdx[cell+j*Nc];
        dir = faceDir[cell+j*Nc];

        d = dir*delta[idx];

        // if   dir == 1: p = p0
        // elif dir ==-1: p = p1
        // elif dir == 0: p = 0
        p = 0.5f*dir*((dir+1.0f)*proj0[idx]+(dir-1.0f)*proj1[idx]);

        dh = max(dh,d); dl = min(dl,d);
        ph = max(ph,p); pl = min(pl,p);
    }

    // minmod
    limiter[cell] = min(1.0f,min(dh/ph, dl/pl));
}


/*
 * First step to assemble the momentum matrix.
 * Compute the diffusion coefficient for each face.
 */
__kernel void assembleMomMat1(uint Nf,
                              float mu,
                              const __global float* alpha0,
                              const __global float* alpha1,
                              __global float* diffCoeff)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    diffCoeff[face] = mu/(alpha0[face]-alpha1[face]);
}


/*
 * Second step to assemble the momentum matrix.
 * Assign a floating-point code to faces according to their type
 * (internal, Dirichlet, or Neuman). This is later used to determine
 * if a face contributes to the off-diagonal and diagonal (internal),
 * just the latter (Dirichlet), or none (Neuman), without using if's.
 */
__kernel void assembleMomMat2(uint Nf,
                              uint Nb,
                              const __global int* bndType,
                              __global float* faceType)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    // default internal
    float ftype = 1.0f;

    if(face<Nb)
    {
        int btype = bndType[face];

        ftype = (btype==INLET || btype==WALL)? -1.0f : 0.0f;
    }

    faceType[face] = ftype;
}


/*
 * Third step to assemble the momentum matrix.
 * Off diagonal and diagonal coefficients, including relaxation.
 */
__kernel void assembleMomMat3(uint Nc,
                              uint width,
                              float relax,
                              const __global int* faceIdx,
                              const __global float* faceDir,
                              const __global float* faceType,
                              const __global float* diffCoeff,
                              const __global float* mdot,
                              __global float* offDiag,
                              __global float* diagonal)
{
    uint cell = get_global_id(0);

    if(cell>=Nc) return;

    // face index, direction, and type
    int idx; float dir, type;

    // diagonal and off-diagonal coefficients
    float a0 = 0.0f, anb = 0.0f;

    uint numel = Nc*width;
    for(uint i=cell; i<numel; i+=Nc)
    {
        idx  = faceIdx[i];
        dir  = faceDir[i];
        type = faceType[idx];

        // the dir*dir mechanism is used to cancel padding faces
        anb = dir*dir*(min(0.0f,dir*mdot[idx])-diffCoeff[idx]);

        // similarly only internal (type=1) and Dirichlet
        // (type=-1) faces contribute to the diagonal
        dir = type*type;
        a0 -= dir*anb;

        // finally only internal faces set the off-diagonal
        dir = 0.5f*(dir+type);
        offDiag[i] = dir*anb;
    }

    diagonal[cell] = a0/relax;
}


/*
 * Compute corrective (2nd order) and boundary momentum fluxes
 */
#define DOT(U,V) (U[0]*V[0] + U[1]*V[1] + U[2]*V[2])
__kernel void comp2ndOrderFlux(uint Nc,
                               uint Nf,
                               uint Nb,
                               uint var,
                               const __global int* connect,
                               const __global int* bndType,
                               const __global float* bndValues,
                               const __global float* geo_r0,
                               const __global float* geo_r1,
                               const __global float* geo_d0,
                               const __global float* geo_d1,
                               const __global float* diffCoeff,
                               const __global float* mdot,
                               const __global float* gradient,
                               const __global float* limiter,
                               __global float* flux)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0, C1;
    float d, m, l0, l1, f, r0[3], r1[3], d0[3], d1[3], g0[3], g1[3];

    // fetch face and "inner side" data
    d = diffCoeff[face];
    m = mdot[face];

    r0[0] = geo_r0[face];      d0[0] = geo_d0[face];
    r0[1] = geo_r0[face+Nf];   d0[1] = geo_d0[face+Nf];
    r0[2] = geo_r0[face+2*Nf]; d0[2] = geo_d0[face+2*Nf];

    C0 = connect[face];
    l0 = limiter[C0];
    g0[0] = gradient[C0];
    g0[1] = gradient[C0+Nc];
    g0[2] = gradient[C0+2*Nc];

    // compute
    f = 0.0f;

    if(face<Nb)
    {
        C1 = bndType[face];
        if(C1==INLET || C1==WALL)
          f = bndValues[face+var*Nb]*(d-m)-d*DOT(g0,d0);
    }
    else
    {
        // fetch "outer side" data
        r1[0] = geo_r1[face];      d1[0] = geo_d1[face];
        r1[1] = geo_r1[face+Nf];   d1[1] = geo_d1[face+Nf];
        r1[2] = geo_r1[face+2*Nf]; d1[2] = geo_d1[face+2*Nf];

        C1 = connect[face+Nf];
        l1 = limiter[C1];
        g1[0] = gradient[C1];
        g1[1] = gradient[C1+Nc];
        g1[2] = gradient[C1+2*Nc];

        f = -max(m,0.0f)*l0*DOT(g0,r0)
            -min(m,0.0f)*l1*DOT(g1,r1)
            -d*(DOT(g0,d0)-DOT(g1,d1));
    }

    flux[face] = f;
}
#undef DOT


/*
 * First step in computing the momentum source term.
 * Initialize it with the relaxation and pressure gradient parts.
 */
__kernel void computeMomSrc1(uint Nc,
                             uint dim,
                             float relax,
                             const __global float* diagonal,
                             const __global float* phi,
                             const __global float* gradient,
                             const __global float* volume,
                             __global float* source)
{
    uint cell = get_global_id(0);

    if(cell<Nc)
      source[cell] = (1.0f-relax)*diagonal[cell]*phi[cell]-
                     volume[cell]*gradient[cell+dim*Nc];
}


/*
 * First step to assemble the pressure correction matrix.
 * Compute the diffusion coefficient for each face.
 */
__kernel void assemblePCmat1(uint Nf,
                             uint Nb,
                             float rho,
                             const __global int* connect,
                             const __global float* alpha0,
                             const __global float* alpha1,
                             const __global float* volume,
                             const __global float* momDiag,
                             __global float* diffCoeff)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0 = connect[face];
    int C1 = connect[face+Nf];
    float coeff;

    if(face<Nb)
        coeff = volume[C0]/momDiag[C0];
    else
        coeff = (volume[C0]+volume[C1])/(momDiag[C0]+momDiag[C1]);

    diffCoeff[face] = rho*coeff/(alpha0[face]-alpha1[face]);
}


/*
 * Second step to assemble the pressure correction matrix.
 * Similar to 2nd momentum step, but for PC outlets are
 * Dirichlet boundaries.
 */
__kernel void assemblePCmat2(uint Nf,
                            uint Nb,
                            const __global int* bndType,
                            __global float* faceType)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    // default internal
    float ftype = 1.0f;

    if(face<Nb)
    {
        int btype = bndType[face];

        ftype = (btype==OUTLET)? -1.0f : 0.0f;
    }

    faceType[face] = ftype;
}


/*
 * Third step to assemble the pressure correction matrix.
 * Off diagonal and diagonal coefficients, including relaxation.
 */
__kernel void assemblePCmat3(uint Nc,
                             uint width,
                             const __global int* faceIdx,
                             const __global float* faceDir,
                             const __global float* faceType,
                             const __global float* diffCoeff,
                             __global float* offDiag,
                             __global float* diagonal)
{
    uint cell = get_global_id(0);

    if(cell>=Nc) return;

    // face index, direction, and type
    int idx; float dir, type;

    // diagonal and off-diagonal coefficients
    float a0 = 0.0f, anb = 0.0f;

    uint numel = Nc*width;
    for(uint i=cell; i<numel; i+=Nc)
    {
        idx  = faceIdx[i];
        dir  = faceDir[i];
        type = faceType[idx];

        // the dir*dir mechanism is used to cancel padding faces
        anb = dir*dir*diffCoeff[idx];

        // similarly only internal (type=1) and Dirichlet
        // (type=-1) faces contribute to the diagonal
        dir = type*type;
        a0 -= dir*anb;

        // finally only internal faces set the off-diagonal
        dir = 0.5f*(dir+type);
        offDiag[i] = dir*anb;
    }

    diagonal[cell] = a0;
}


/*
 * Compute mass flows
 */
#define DOT(U,V) (U[0]*V[0] + U[1]*V[1] + U[2]*V[2])
__kernel void computeMassFlows(uint Nc,
                               uint Nf,
                               uint Nb,
                               float rho,
                               const __global int* connect,
                               const __global int* bndType,
                               const __global float* bndValues,
                               const __global float* geo_wf,
                               const __global float* geo_r0,
                               const __global float* geo_r1,
                               const __global float* geo_Af,
                               const __global float* diffCoeff,
                               const __global float* u,
                               const __global float* v,
                               const __global float* w,
                               const __global float* p,
                               const __global float* gradP,
                               __global float* mdot)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0, C1 = -1;
    float wf, df, mf, rf[3], Af[3], Vf[3], Gf[3];

    // fetch face and "inner side" data
    C0 = connect[face];

    wf = geo_wf[face];
    df = diffCoeff[face];

    Af[0] = geo_Af[face]; Af[1] = geo_Af[face+Nf]; Af[2] = geo_Af[face+2*Nf];

    // start defining the "center to center" vector
    rf[0] = geo_r0[face]; rf[1] = geo_r0[face+Nf]; rf[2] = geo_r0[face+2*Nf];

    // start the interpolation of nabla(P) at face
    Gf[0] = wf*gradP[C0]; Gf[1] = wf*gradP[C0+Nc]; Gf[2] = wf*gradP[C0+2*Nc];

    if(face<Nb)
    {
        Vf[0] = bndValues[face];
        Vf[1] = bndValues[face+Nb];
        Vf[2] = bndValues[face+2*Nb];

        mf = rho*DOT(Af,Vf);

        // only outlets get added dissipation
        if(bndType[face]==OUTLET)
            mf += df*(p[C0]+DOT(Gf,rf)-bndValues[face+3*Nb]);
    }
    else
    {
        C1 = connect[face+Nf];

        // linear interpolation of velocities
        Vf[0] = wf*u[C0]+(1.0f-wf)*u[C1];
        Vf[1] = wf*v[C0]+(1.0f-wf)*v[C1];
        Vf[2] = wf*w[C0]+(1.0f-wf)*w[C1];

        // update "center to center" vector
        rf[0] -= geo_r1[face];
        rf[1] -= geo_r1[face+Nf];
        rf[2] -= geo_r1[face+2*Nf];

        // complete the interpolation of nabla(P) at face
        Gf[0] += (1.0f-wf)*gradP[C1];
        Gf[1] += (1.0f-wf)*gradP[C1+Nc];
        Gf[2] += (1.0f-wf)*gradP[C1+2*Nc];

        mf = rho*DOT(Af,Vf) + df*(p[C0]-p[C1]+DOT(Gf,rf));
    }

    mdot[face] = mf;
}
#undef DOT


/*
 * Correct mass flows
 */
__kernel void correctMassFlows(uint Nf,
                               uint Nb,
                               const __global int* connect,
                               const __global int* bndType,
                               const __global float* diffCoeff,
                               const __global float* pc,
                               __global float* mdot)
{
    uint face = get_global_id(0);

    if(face>=Nf) return;

    int C0 = connect[face];
    int C1 = connect[face+Nf];

    float pc_C1;

    if(face<Nb)
        pc_C1 = (bndType[face]==OUTLET)? 0.0f : pc[C0];
    else
        pc_C1 = pc[C1];

    mdot[face] -= diffCoeff[face]*(pc_C1-pc[C0]);
}


/*
 * First step to correct cell velocities.
 * Set pressure correction at boundary faces.
 */
__kernel void boundaryValuesPC(uint Nb,
                               const __global int* bndType,
                               const __global int* cellIdx,
                               const __global float* pc,
                               __global float* pcb)
{
    uint face = get_global_id(0);

    if(face>=Nb) return;

    int type = bndType[face];
    int cell = cellIdx[face];

    pcb[face+4*Nb] = (type==OUTLET)? 0.0f : pc[cell];
}


/*
 * Last step to correct cell velocities.
 * The pressure correction gradient should not be divided by volume.
 */
__kernel void correctVelocity(uint Nc,
                              const __global float* momDiag,
                              const __global float* gradPC,
                              __global float* u,
                              __global float* v,
                              __global float* w)
{
    uint cell = get_global_id(0);

    if(cell>=Nc) return;

    float d = 1.0f/momDiag[cell];

    u[cell] -= d*gradPC[cell];
    v[cell] -= d*gradPC[cell+Nc];
    w[cell] -= d*gradPC[cell+2*Nc];
}


/*
 * SET kernel. Initialize an array with a constant value.
 */
__kernel void set(uint n, float a, __global float* x)
{
    uint i = get_global_id(0);

    if(i<n) x[i] = a;
}


/*
 * COPY kernel. Strides not supported.
 */
__kernel void copy(uint n,
                   const __global float* x,
                   __global float* y)
{
    uint i = get_global_id(0);

    if(i<n) y[i] = x[i];
}


/*
 * AXPY kernel. Strides not supported.
 */
__kernel void axpy(uint n,
                   float a,
                   const __global float* x,
                   __global float* y)
{
    uint i = get_global_id(0);

    if(i<n) y[i] += a*x[i];
}


/*
 * Dot product. Final reduction needs to be done by Host.
 */
__kernel void dotp(int szVec,
                   __global const float* vecA,
                   __global const float* vecB,
                   __global float* wrkGrpRes,
                   __local  float* cache)
{
    int iglb = get_global_id(0);
    int iloc = get_local_id(0);
    int igrp = get_group_id(0);
    int stride = get_local_size(0); // must be power of two and size(cache) = local_size

    // store the element-wise multiplications in the cache and pad it
    if(iglb < szVec)
        cache[iloc] = vecA[iglb]*vecB[iglb];
    else
        cache[iloc] = 0.f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // add the products of this work group
    while(stride > 1)
    {
        stride >>= 1;
        if(iloc < stride) cache[iloc] += cache[iloc+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write the result of this work group
    if(iloc == 0) wrkGrpRes[igrp] = cache[0];
}


/*
 * Generic matrix-vector multiplication.
 * Matrix in LIL format, transpose product, strides, and offsets not supported.
 * Used to sum fluxes and during solution of linear systems.
 */
__kernel void gemv(uint n,
                   uint m,
                   float alpha,
                   float beta,
                   const __global int* acols,
                   const __global float* avals,
                   const __global float* x,
                   __global float* y)
{
    uint i = get_global_id(0);

    if(i>=n) return;

    float sum = 0.0f;
    uint k = i;

    for(uint j=0; j<m; ++j)
    {
        sum += avals[k]*x[acols[k]];
        k += n;
    }

    y[i] = alpha*sum + beta*y[i];
}


/*
 * Banded matrix-vector multiplication.
 * Matrix restricted to main diagonal.
 * Transpose product, strides, and offsets not supported.
 */
__kernel void gbmv(uint n,
                   float alpha,
                   float beta,
                   const __global float* diag,
                   const __global float* x,
                   __global float* y)
{
    uint i = get_global_id(0);

    if(i<n) y[i] = alpha*diag[i]*x[i] + beta*y[i];
}


/*
 * Scale the rows of a matrix (diagonal preconditioner).
 */
__kernel void scl(uint n,
                  uint m,
                  const __global float* scales,
                  __global float* mat)
{
    uint i = get_global_id(0);

    if(i>=n) return;

    float scale = 1.0f/scales[i];

    for(uint j=0; j<m; ++j) mat[i+j*n] *= scale;
}


/*
 * Sparse Approximate Inverse.
 * Symmetric matrix in LIL format, with separately stored diagonal.
 */
#define MAXSIZE 8
#define mat(i,j) _mat[(i)*MAXSIZE+(j)]

__kernel void spai(int nRows,
                   int nColsMax,
                   const __global int* nCols,
                   const __global int* colIdx,
                   const __global float* aii,
                   const __global float* aij,
                   __global float* mii,
                   __global float* mij)
{
    int row = get_global_id(0);

    // local matrices in row-major format (determined by "mat" macro),
    // global matrices are column-major

    // matrix and right hand side (and result) of the least-squares problem
    __private float _mat[MAXSIZE*MAXSIZE];
    __private float vec[MAXSIZE];

    // since mat is also used to store its factorization,
    // the diagonal factors need to be stored separately
    __private float diag[MAXSIZE];

    if(row>=nRows) return;

    int cols = nCols[row];

    int i, j, k, ii, jj, iptr, icol, jptr, jcol, iiptr, iicol, jjptr, jjcol;
    float sum;

    // initialize result vector
    for(i=0; i<=nColsMax; ++i) vec[i] = 0.0f;

    // set rhs and build matrix for the normal equations
    for(i=0; i<=cols; ++i)
    {
        if(i<cols)
        {
            iptr = i*nRows+row;
            icol = colIdx[iptr];
            vec[i] = aij[iptr];
        }
        else
        {
            icol = row;
            vec[i] = aii[icol];
        }

        // diagonal
        sum = aii[icol]*aii[icol];
        for(ii=0; ii<nColsMax; ++ii)
        {
            iiptr = ii*nRows+icol;
            sum += aij[iiptr]*aij[iiptr];
        }
        mat(i,i) = sum;

        // upper triangular part
        for(j=i+1; j<=cols; ++j)
        {
            if(j<cols)
            {
                jptr = j*nRows+row;
                jcol = colIdx[jptr];
            }
            else
            {
                jcol = row;
            }

            // naive way of doing the sparse dot products between rows
            // assuming unique unsorted indices
            sum = 0.0f;
            for(ii=0; ii<nColsMax; ++ii)
            {
                iiptr = ii*nRows+icol;
                iicol = colIdx[iiptr];

                for(jj=0; jj<nColsMax; ++jj)
                {
                    jjptr = jj*nRows+jcol;
                    jjcol = colIdx[jjptr];

                    if(iicol==jjcol) {sum += aij[iiptr]*aij[jjptr]; break;}
                }
            }
            for(jj=0; jj<nColsMax; ++jj)
            {
                jjptr = jj*nRows+jcol;
                jjcol = colIdx[jjptr];

                if(icol==jjcol) {sum += aii[icol]*aij[jjptr]; break;}
            }
            for(ii=0; ii<nColsMax; ++ii)
            {
                iiptr = ii*nRows+icol;
                iicol = colIdx[iiptr];

                if(iicol==jcol) {sum += aij[iiptr]*aii[jcol]; break;}
            }
            mat(i,j) = sum;
        }
    }

    // Cholesky factorization, stored in lower part of matrix
    for(i=0; i<=cols; i++)
    {
        // diagonal coefficient
        sum = mat(i,i);
        for(k=i-1; k>=0; k--) sum -= mat(i,k)*mat(i,k);
        diag[i] = sqrt(sum);

        // off-diagonal (lower) terms
        for(j=i+1; j<=cols; j++)
        {
            sum = mat(i,j);
            for(k=i-1; k>=0; k--) sum -= mat(i,k)*mat(j,k);
            mat(j,i) = sum/diag[i];
        }
    }

    // Triangular solves
    // rhs overwritten first with "y" then with "x"
    for(i=0; i<=cols; i++)
    {
        sum = vec[i];
        for(k=i-1; k>=0; k--) sum -= mat(i,k)*vec[k];
        vec[i] = sum/diag[i];
    }
    for(i=cols; i>=0; i--)
    {
        sum = vec[i];
        for(k=i+1; k<=cols; k++) sum -= mat(k,i)*vec[k];
        vec[i] = sum/diag[i];
    }

    // write result
    mii[row] = vec[cols];
    vec[cols] = 0.0f;
    for(i=0; i<nColsMax; ++i) mij[i*nRows+row] = vec[i];
}
