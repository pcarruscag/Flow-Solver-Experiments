//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef TURBULENCEMODELS_H
#define TURBULENCEMODELS_H

#include <Eigen/Dense>

using namespace Eigen;

namespace flow
{
namespace turbulenceModels
{
// All turbulence models will implement this interface
class ModelBase
{
  public:
    enum ModelOptions {LAMINAR,
                       SST};
    enum ModelVariables {FIRST,
                         SECOND};
    enum FirstVariableWallBC {DIRICHLET,
                              NEUMAN};
  protected:
    FirstVariableWallBC m_var1wallBC;
    float m_max1, m_max2;

  public:
    static ModelBase* makeModel(const ModelOptions option);

    inline ModelBase(FirstVariableWallBC bc) : m_var1wallBC(bc), m_max1(0.f), m_max2(0.f) {}
    virtual int setup(const float rho, const float mu,
                      const VectorXf*  v1_ptr,     const VectorXf*  v2_ptr,
                      const MatrixX3f* v1grad_ptr, const MatrixX3f* v2grad_ptr,
                      const MatrixX3f* uGrad_ptr,  const MatrixX3f* vGrad_ptr,
                      const MatrixX3f* wGrad_ptr,  const VectorXf*  wallDist_ptr,
                      const VectorXf*  mut_ptr) = 0;
    virtual void updateInternalData() = 0;
    virtual void turbulentViscosity(const float relax, VectorXf& mut) const = 0;
    virtual void rhsSourceTerm(const ModelVariables variable, VectorXf& source) const = 0;
    virtual void linearizedSourceTerm(const ModelVariables variable, const int* diag_ptr,
                                      const float* vol, float* matrix) const = 0;
    virtual float viscosityMultiplier (const ModelVariables variable, const int cell) const = 0;
    virtual Matrix<float,1,5> wallValues(const int cell, const float Up,
                                         float &var1, float &var2) const = 0;
    virtual ~ModelBase() {};

    inline FirstVariableWallBC firstVarWallBC() const {return m_var1wallBC;}
    inline float maxValueOf(const ModelVariables variable) const
    {
        return (variable==FIRST)? m_max1 : m_max2;
    }
};

class LaminarFlow: public ModelBase
{
  public:
    inline LaminarFlow() : ModelBase(DIRICHLET) {}
    inline int setup (const float, const float, const VectorXf*, const VectorXf*,
                      const MatrixX3f*, const MatrixX3f*, const MatrixX3f*, const MatrixX3f*,
                      const MatrixX3f*, const VectorXf*, const VectorXf*) {return 0;}
    inline void updateInternalData() {}
    inline void turbulentViscosity(const float relax, VectorXf& mut) const {mut *= (1.0f-relax);}
    inline void rhsSourceTerm(const ModelVariables, VectorXf&) const {}
    inline void linearizedSourceTerm(const ModelVariables, const int*, const float*, float*) const {}
    inline float viscosityMultiplier (const ModelVariables, const int) const {return 0.0f;}
    inline Matrix<float,1,5> wallValues(const int, const float, float&, float&) const
    {
        return Matrix<float,1,5>::Zero();
    }
};

class MenterSST: public ModelBase
{
  private:
    bool m_modelReady;

    // Blending function and strain magnitude (squared)
    VectorXf m_F1, m_Ssqr;

    // Model constants
    static constexpr float m_smallValue{1e-10},
        m_CT{0.60}, m_a1{0.31}, m_kappa{0.41},
        m_betaStar{0.09}, m_alphaStar{1.0},
        m_beta1{0.075},  m_beta2{0.0828},
        m_sigmaK1{0.85}, m_sigmaK2{1.0},
        m_sigmaW1{0.50}, m_sigmaW2{0.856},
        m_gamma1{m_beta1/m_betaStar-m_sigmaW1*m_kappa*m_kappa/std::sqrt(m_betaStar)},
        m_gamma2{m_beta2/m_betaStar-m_sigmaW2*m_kappa*m_kappa/std::sqrt(m_betaStar)};

    // Fluid properties
    float m_rho, m_mu;

    // Pointers to required flow field and geometric data
    int m_nCells;
    const MatrixX3f *m_uGrad_ptr, *m_vGrad_ptr, *m_wGrad_ptr;
    const VectorXf  *m_k_ptr, *m_o_ptr;
    const MatrixX3f *m_kGrad_ptr, *m_oGrad_ptr;
    const VectorXf  *m_wallDist_ptr, *m_mut_ptr;

  public:
    MenterSST();
    int setup(const float rho, const float mu,
              const VectorXf*  v1_ptr,     const VectorXf*  v2_ptr,
              const MatrixX3f* v1grad_ptr, const MatrixX3f* v2grad_ptr,
              const MatrixX3f* uGrad_ptr,  const MatrixX3f* vGrad_ptr,
              const MatrixX3f* wGrad_ptr,  const VectorXf*  wallDist_ptr,
              const VectorXf*  mut_ptr);
    void updateInternalData();
    void turbulentViscosity(const float relax, VectorXf& mut) const;
    void rhsSourceTerm(const ModelVariables variable, VectorXf& source) const;
    void linearizedSourceTerm(const ModelVariables variable, const int* diag_ptr,
                              const float* vol, float* matrix) const;
    inline float viscosityMultiplier(const ModelVariables variable, const int cell) const
    {
        return m_F1(cell) *((variable==FIRST)? m_sigmaK1 : m_sigmaW1)+
         (1.0f-m_F1(cell))*((variable==FIRST)? m_sigmaK2 : m_sigmaW2);
    }
    Matrix<float,1,5> wallValues(const int cell, const float Up, float &var1, float &var2) const;
};
}
}
#endif // TURBULENCEMODELS_H
