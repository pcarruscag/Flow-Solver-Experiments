//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef SPLINES_H
#define SPLINES_H

#include <vector>

namespace mathUtils
{
    namespace interpolation
    {
        template <typename type> class Cspline
        {
          public:
            Cspline();
            Cspline(const std::vector<type> &x, const std::vector<type> &y, const double endCondition=0.0);
            void setData(const std::vector<type> &x, const std::vector<type> &y, const double endCondition=0.0);
            type operator()(const type x) const;
            std::vector<type> operator()(const std::vector<type> &x) const;
            type deriv(const type x) const;
            std::vector<type> deriv(const std::vector<type> &x) const;
          private:
            bool m_dataIsReady;
            double m_extrapTol = 0.1; //10% of adjacent interval
            int m_nPt;
            std::vector<type> m_x, m_y, m_dx, m_d2y;
            int m_getIndex(const type &x) const;
            type m_evaluate(const type &x) const;
            type m_deriv(const type &x) const;
        };

        template <typename type> class PiecewiseLinear
        {
          public:
            PiecewiseLinear();
            PiecewiseLinear(const std::vector<type> &x, const std::vector<type> &y);
            void setData(const std::vector<type> &x, const std::vector<type> &y);
            type operator()(const type x) const;
            std::vector<type> operator()(const std::vector<type> &x) const;
          private:
            bool m_dataIsReady;
            double m_extrapTol = 0.1; //10% of adjacent interval
            int m_nPt;
            std::vector<type> m_x, m_y;
            int m_getIndex(const type &x) const;
            type m_evaluate(const type &x) const;
        };

        template <typename type> std::vector<type> splineInterp(const std::vector<type> &x,
                                                                const std::vector<type> &y,
                                                                const std::vector<type> &xi,
                                                                const double endCondition=0.0);

        template <typename type> std::vector<type> pwiseLinInterp(const std::vector<type> &x,
                                                                const std::vector<type> &y,
                                                                const std::vector<type> &xi);
    }
}
#endif //SPLINES_H
