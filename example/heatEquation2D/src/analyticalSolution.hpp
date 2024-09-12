/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>

ALPAKA_FN_HOST auto analyticalSolution(double const x, double const y, double const t) -> double
{
    constexpr double pi = alpaka::math::constants::pi;
    return std::exp(-pi * pi * t) * (std::sin(pi * x) + std::sin(pi * y));
}

//! Valdidate calculated solution in the buffer to the analytical solution at t=tMax
//!
//! \param buffer buffer holding the solution at t=tMax
//! \param extent extents of the buffer
//! \param dx
//! \param dy
//! \param t
template<typename T_Buffer, typename T_Extent>
ALPAKA_FN_HOST auto validateSolution(
    T_Buffer const& buffer,
    T_Extent const& extent,
    double const dx,
    double const dy,
    double const t) -> std::pair<bool, double>
{
    // Calculate error
    double maxError = 0.0;
    for(uint32_t j = 1; j < extent[0] - 1; ++j)
    {
        for(uint32_t i = 0; i < extent[1]; ++i)
        {
            auto const error = std::abs(buffer.data()[j * extent[1] + i] - analyticalSolution(i * dx, j * dy, t));
            maxError = std::max(maxError, error);
        }
    }

    constexpr double errorThreshold = 1e-5;
    return std::make_pair(maxError < errorThreshold, maxError);
}
