/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>

//! Exact solution to the test problem
//! u_t(x, y, t) = u_xx(x, t) + u_yy(y, t), x in [0, 1], y in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//! u(0, y) = sin(pi * y)
//!
//! \param x value of x
//! \param x value of y
//! \param t value of t

ALPAKA_FN_HOST_ACC auto exactSolution(double const x, double const y, double const t) -> double
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    return std::exp(-pi * pi * t) * std::sin(pi * x) + std::exp(-pi * pi * t) * std::sin(pi * y);
}

//! Valdidate calculated solution in the buffer to the analytical solution at t=tMax
//!
//! \param buffer buffer holding the solution at t=tMax
//! \param extent extents of the buffer
//! \param dx
//! \param dy
//! \param tMax time at simulation end

template<typename T_Buffer, typename T_Extent>
ALPAKA_FN_HOST_ACC auto validateSolution(
    T_Buffer const& buffer,
    T_Extent const& extent,
    double const dx,
    double const dy,
    double const tMax) -> std::pair<bool, double>
{
    // Calculate error
    double maxError = 0.0;
    for(uint32_t j = 1; j < extent[0] - 1; ++j)
    {
        for(uint32_t i = 0; i < extent[1]; ++i)
        {
            auto const error = std::abs(buffer.data()[j * extent[1] + i] - exactSolution(i * dx, j * dy, tMax));
            maxError = std::max(maxError, error);
        }
    }

    constexpr double errorThreshold = 1e-5;
    return std::make_pair(maxError < errorThreshold, maxError);
}

//! Initialize the buffer to the analytical solution at t=0
//!
//! \param buffer buffer holding the solution at tMax
//! \param extent extents of the buffer
//! \param dx
//! \param dy

template<typename T_Buffer, typename T_Extent>
ALPAKA_FN_HOST_ACC auto initalizeBuffer(T_Buffer& buffer, T_Extent const& extent, double const dx, double const dy)
    -> void
{
    // Apply initial conditions for the test problem
    for(uint32_t j = 0; j < extent[0]; ++j)
    {
        for(uint32_t i = 0; i < extent[1]; ++i)
        {
            buffer.data()[j * extent[1] + i] = exactSolution(i * dx, j * dy, 0.0);
        }
    }
}
