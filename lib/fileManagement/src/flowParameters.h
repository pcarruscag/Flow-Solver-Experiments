//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef FLOWPARAMETERS_H
#define FLOWPARAMETERS_H

#include "../../flow/src/turbulenceModels.h"
#include <string>
#include <fstream>
#include <sstream>

using std::ifstream;
using std::stringstream;
using std::string;

namespace flow
{
    class PressureBasedCoupledSolver;
    class PressureBasedSegregatedSolver;
}

namespace fileManagement
{
    class FlowParamManager
    {
        friend class flow::PressureBasedCoupledSolver;
        friend class flow::PressureBasedSegregatedSolver;
        friend void _flowParametersTester();
      public:
        enum InletVariable {VELOCITY = 1,
                            MASSFLOW = 2,
                            TOTALPRESSURE = 3};
        enum FlowDirection {NORMAL = 1,
                            DIRECTION = 2,
                            COMPONENTS = 3};
        enum CoordSysType  {CARTESIAN = 1,
                            CYLINDRICAL = 2};
        enum TurbSpecType  {INTENSITY = 1,
                            KOMEGA = 2};
        struct inletConditions
        {
            InletVariable variable;
            FlowDirection direction;
            CoordSysType  coordinate;
            TurbSpecType  turbSpec;
            float scalar;
            float components[3];
            float turbVal1;
            float turbVal2;
        };
        struct outletConditions
        {
            float pressure;
            bool  massFlowOption;
            float massFlow;
        };
        struct domainConditions
        {
            float rotationalSpeed;
            bool  rotatingShroud;
            bool  rotatingHub;
        };
        struct fluidProperties
        {
            float rho;
            float mu;
        };
        struct controlParameters
        {
            flow::turbulenceModels::ModelBase::ModelOptions turbulenceType;
            float mapTol;
            float venkatK;
            float mdotRelax;
            float cflNumber;
            float relax0,   relax1;
            float relax0_t, relax1_t;
            float initViscRatio;
            float tol;
            int minIt, maxIt;
            float linSolTol;
            int linSolMaxIt;
            float bigNumber;
        };
        FlowParamManager();
        int readFile(const string &filePath);
        bool dataIsReady() const;
      private:
        bool m_dataIsReady;
        string m_filePath;

        inletConditions  m_inletConditions;
        outletConditions m_outletConditions;
        domainConditions m_domainConditions;
        fluidProperties  m_fluidProperties;
        controlParameters m_controlParams;

        string m_inletHeader  = "### Inlet Conditions ###";
        string m_outletHeader = "### Outlet Conditions ###";
        string m_domainHeader = "### Domain Conditions ###";
        string m_fluidPropHeader = "### Fluid Properties ###";
        string m_fluidModelHeader = "### Fluid Models ###";
        string m_convergenceHeader = "### Convergence Parameters ###";
        string m_advancedHeader = "### Advanced Parameters ###";

        inline void m_getLineFromFile(ifstream& file, stringstream& stream) const;
    };

    inline stringstream& operator>>(stringstream& strStream, FlowParamManager::InletVariable& enumVar) {
        int intVar = 0;
        if (strStream >> intVar)
            enumVar = static_cast<FlowParamManager::InletVariable>(intVar);
        return strStream;
    }
    inline stringstream& operator>>(stringstream& strStream, FlowParamManager::FlowDirection& enumVar) {
        int intVar = 0;
        if (strStream >> intVar)
            enumVar = static_cast<FlowParamManager::FlowDirection>(intVar);
        return strStream;
    }
    inline stringstream& operator>>(stringstream& strStream, FlowParamManager::CoordSysType& enumVar) {
        int intVar = 0;
        if (strStream >> intVar)
            enumVar = static_cast<FlowParamManager::CoordSysType>(intVar);
        return strStream;
    }
    inline stringstream& operator>>(stringstream& strStream, FlowParamManager::TurbSpecType& enumVar) {
        int intVar = 0;
        if (strStream >> intVar)
            enumVar = static_cast<FlowParamManager::TurbSpecType>(intVar);
        return strStream;
    }
}

#endif // FLOWPARAMETERS_H
