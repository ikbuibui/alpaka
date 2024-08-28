/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

template<typename T, typename U>
using const_match = std::conditional_t<std::is_const_v<T>, U const, U>;

//! Helper function to get a pointer to an element in a multidimensional buffer
//!
//! \tparam T type of the element
//! \tparam TDim dimension of the buffer
//! \tparam TIdx index type
//! \param ptr pointer to the buffer
//! \param idx index of the element
//! \param pitch pitch of the buffer
template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T* getElementPtr(T* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T*>(
        reinterpret_cast<const_match<T, std::byte>*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}

template<typename TNodes, typename TChunk>
auto isValidChunking(TNodes _numNodes, TChunk _chunkSize) -> bool
{
    if(!(_numNodes[0] % _chunkSize[0] == 0 && _numNodes[1] % _chunkSize[1] == 0))
    {
        std::cerr << "Domain must be divisible by chunk size \n";
        return false;
    }
    else
    {
        return true;
    }
}

// Check the stability condition
auto isStable(double _dx, double _dy, double _dt) -> bool
{
    double r = _dt / std::min(_dx * _dx, _dy * _dy);
    if(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return false;
    }
    else
    {
        return true;
    }
}
