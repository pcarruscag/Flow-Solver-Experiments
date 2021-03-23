//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef ADJOINTPARAMETERS_H
#define ADJOINTPARAMETERS_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using std::ifstream;
using std::stringstream;
using std::string;
using std::vector;

namespace fileManagement
{
    class AdjointParamManager
    {
      private:
        // data read from the input file
        vector<int> m_objectives;
        int    m_maxIt;
        double m_tolerance;
        double m_alphaMin;
        double m_alphaMax;
        bool   m_firstOrder;
        bool   m_rightPrec;
        int    m_linSolMaxIt;
        double m_linSolMinTol;
        double m_linSolInitTol;
        double m_linSolRatioTol;
        string m_scratchDir;

        // definitions used to process the file
        string m_objectivesHeader  = "### Objectives ###";
        string m_convergenceHeader = "### Convergence Parameters ###";
        string m_advancedHeader    = "### Advanced Parameters ###";
        vector<string> m_objectiveNames;

        string m_filePath;
        bool m_dataIsReady;
      public:
        AdjointParamManager() {
            // same order as the enumerator in the solver class
            m_objectiveNames.resize(11);
            m_objectiveNames[0] = "MDOT_IN";
            m_objectiveNames[1] = "MDOT_OUT";
            m_objectiveNames[2] = "BLOCKAGE";
            m_objectiveNames[3] = "THRUST";
            m_objectiveNames[4] = "TORQUE";
            m_objectiveNames[5] = "DELTAP_TT";
            m_objectiveNames[6] = "DELTAP_TS";
            m_objectiveNames[7] = "DELTAP_SS";
            m_objectiveNames[8] = "ETA_TT";
            m_objectiveNames[9] = "ETA_TS";
            m_objectiveNames[10]= "ETA_SS";
            m_dataIsReady = false;
        }
        int readFile(const string &filePath);
        bool dataIsReady() const {return m_dataIsReady;}
        int getObjectives(vector<int>& objectives) const;
        int getFirstOrder(bool& firstOrder) const;
        int getRightPrec(bool& rightPrec) const;
        int getScratchDir(string& scratchDir) const;

        template<typename T>
        int getStopCriteria(int& maxIters, T& tolerance) const
        {
            maxIters  = m_maxIt;
            tolerance = m_tolerance;
            return !m_dataIsReady;
        }

        template<typename T>
        int getRelaxLimits(T& alphaMin, T& alphaMax) const
        {
            alphaMin = m_alphaMin;
            alphaMax = m_alphaMax;
            return !m_dataIsReady;
        }

        template<typename T>
        int getLinSolParam(int& maxIters, T& initTol, T& minTol, T& ratioTol) const
        {
            minTol = m_linSolMinTol;
            initTol = m_linSolInitTol;
            ratioTol = m_linSolRatioTol;
            maxIters = m_linSolMaxIt;
            return !m_dataIsReady;
        }

      private:
        inline void m_getLineFromFile(ifstream& file, stringstream& stream) const;
    };
}
#endif // ADJOINTPARAMETERS_H
