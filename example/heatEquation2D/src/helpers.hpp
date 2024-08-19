/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T* getElementPtr(T* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}

template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T const* getElementPtr(T const* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T const*>(reinterpret_cast<std::byte const*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}
