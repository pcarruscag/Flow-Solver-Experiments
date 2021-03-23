//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "turbulenceModels.h"

#include <omp.h>
#include <cassert>
#include <math.h>
#include <algorithm>
#include <iostream>

namespace flow
{
namespace turbulenceModels
{
ModelBase* ModelBase::makeModel(const ModelOptions option)
{
    switch(option)
    {
      case LAMINAR:
        return new LaminarFlow();
      case SST:
        return new MenterSST();
      default:
        return nullptr;
    }
}

MenterSST::MenterSST() : ModelBase(NEUMAN), m_modelReady(false) {}

int MenterSST::setup(const float rho, const float mu,
                  const VectorXf*  v1_ptr,     const VectorXf*  v2_ptr,
                  const MatrixX3f* v1grad_ptr, const MatrixX3f* v2grad_ptr,
                  const MatrixX3f* uGrad_ptr,  const MatrixX3f* vGrad_ptr,
                  const MatrixX3f* wGrad_ptr,  const VectorXf*  wallDist_ptr,
                  const VectorXf*  mut_ptr)
{
    m_modelReady = false;

    if(!v1_ptr || !v2_ptr || !v1grad_ptr || !v2grad_ptr || !mut_ptr ||
       !uGrad_ptr || !vGrad_ptr || !wGrad_ptr  || !wallDist_ptr)
        return 1;
    if(v1_ptr->rows()<1)
        return 1;

    m_rho = rho;  m_mu = mu;
    m_k_ptr = v1_ptr;         m_o_ptr = v2_ptr;
    m_kGrad_ptr = v1grad_ptr; m_oGrad_ptr = v2grad_ptr;
    m_uGrad_ptr = uGrad_ptr;  m_vGrad_ptr = vGrad_ptr;
    m_wGrad_ptr = wGrad_ptr;  m_wallDist_ptr = wallDist_ptr;
    m_mut_ptr   = mut_ptr;

    m_nCells = m_k_ptr->rows();
    m_F1.setConstant(m_nCells,0.5);
    m_Ssqr.setZero(m_nCells);

    m_modelReady = true;

    return 0;
}

void MenterSST::updateInternalData()
{
    using std::max; using std::min; using std::pow; using std::sqrt;
    assert(m_modelReady);

    // Compute S
    #pragma omp simd
    for(int i=0; i<m_nCells; ++i)
        m_Ssqr(i) = 2.0f*(pow((*m_uGrad_ptr)(i,0),2.0f)+
                          pow((*m_vGrad_ptr)(i,1),2.0f)+
                          pow((*m_wGrad_ptr)(i,2),2.0f))+
                          pow((*m_uGrad_ptr)(i,1)+(*m_vGrad_ptr)(i,0),2.0f)+
                          pow((*m_uGrad_ptr)(i,2)+(*m_wGrad_ptr)(i,0),2.0f)+
                          pow((*m_vGrad_ptr)(i,2)+(*m_wGrad_ptr)(i,1),2.0f);

    // Compute blending function F1 and maximum variable values
    m_max1 = m_max2 = 0.0f;
    for(int i=0; i<m_nCells; ++i)
    {
        float y = (*m_wallDist_ptr)(i),
              k = (*m_k_ptr)(i),
              omega = (*m_o_ptr)(i);
        float CDkw = max(float(m_smallValue),
                         m_kGrad_ptr->row(i).dot(m_oGrad_ptr->row(i))/omega);
        float arg1 = min(2.0f*k/(CDkw*pow(y,2.0f)),
                         max(sqrt(k)/(m_betaStar*omega*y),
                             500.0f*m_mu/(m_rho*omega*pow(y,2.0f))));
        m_F1(i) = std::tanh(pow(arg1,4.0f));
        m_max1 = max(m_max1,k);
        m_max2 = max(m_max2,omega);
    }
}

void MenterSST::turbulentViscosity(const float relax, VectorXf& mut) const
{
    using std::max; using std::min; using std::pow; using std::sqrt;
    assert(m_modelReady);

    for(int i=0; i<m_nCells; ++i)
    {
        float y = (*m_wallDist_ptr)(i),
              k = (*m_k_ptr)(i),
              omega = (*m_o_ptr)(i);
        float F2 = std::tanh(pow(max(2.0f*sqrt(k)/(m_betaStar*omega*y),
                                     500.0f*m_mu/(m_rho*omega*pow(y,2.0f))),2.0f));
        float T = min(1.0f/max(omega/m_alphaStar,sqrt(m_Ssqr(i))*F2/m_a1),
                      m_CT/(1.7321f*sqrt(m_Ssqr(i))));
        mut(i) += relax*(m_rho*k*T-mut(i));
    }
}

void MenterSST::rhsSourceTerm(const ModelVariables variable, VectorXf& source) const
{
    assert(m_modelReady);

    if(variable == FIRST) // k: Gk
        source = m_mut_ptr->cwiseProduct(m_Ssqr);
    else // omega: Gw + Dw
        source = m_rho*m_Ssqr.cwiseProduct(m_F1*m_gamma1+(VectorXf::Ones(m_nCells)-m_F1)*m_gamma2)+
                 2.0f*m_rho*m_sigmaW2*(VectorXf::Ones(m_nCells)-m_F1).cwiseProduct(
                 m_kGrad_ptr->cwiseProduct(*m_oGrad_ptr).rowwise().sum()).cwiseQuotient(*m_o_ptr);
}

void MenterSST::linearizedSourceTerm(const ModelVariables variable, const int* diag_ptr,
                                     const float* vol, float* matrix) const
{
    if(variable==FIRST) {
        #pragma omp simd
        for(int i=0; i<m_nCells; i++)
            matrix[diag_ptr[i]] = m_rho*m_betaStar*(*m_o_ptr)(i)*vol[i];
    }
    else {
        #pragma omp simd
        for(int i=0; i<m_nCells; i++)
            matrix[diag_ptr[i]] = m_rho*(m_beta1*m_F1(i)+m_beta2*(1.0f-m_F1(i)))*(*m_o_ptr)(i)*vol[i];
    }
}

Matrix<float,1,5> MenterSST::wallValues(const int cell, const float Up,
                                        float &var1, float &var2) const
{
    using std::sqrt; using std::pow; using std::exp; using std::min;

    float y = (*m_wallDist_ptr)(cell), k = (*m_k_ptr)(cell);

    var1 = (m_var1wallBC == DIRICHLET)? m_smallValue : k;

    float Rey = sqrt(k)*y*m_rho/m_mu, beta = m_beta1*m_F1(cell)+m_beta2*(1.0f-m_F1(cell));
    float g = exp(Rey/-11.0f);
    float uref = sqrt(g*m_mu*Up/(m_rho*y)+(1.0f-g)*sqrt(m_betaStar)*k);

    var2 = g*6.0f*m_mu/(m_rho*beta*pow(y,2.0f))+(1.0f-g)*uref/(sqrt(m_betaStar)*m_kappa*y);

    // return (blending function, wall specific Gk, reference velocity, u+ and y+)
    return (Matrix<float,1,5>() << g, (1.0f-g)*m_rho*Up*pow(uref,2.0f)/y,
                                   uref, Up/uref, y*m_rho*uref/m_mu).finished();
}

}
}
