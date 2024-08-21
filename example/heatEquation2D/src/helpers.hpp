/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

template<typename T, typename U>
using const_match = std::conditional_t<std::is_const_v<T>, U const, U>;

template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T* getElementPtr(T* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T*>(
        reinterpret_cast<const_match<T, std::byte>*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}
