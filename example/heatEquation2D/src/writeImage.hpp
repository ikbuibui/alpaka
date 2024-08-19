/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <pngwriter.h>

#include <cstdint>
#include <iomanip>
#include <sstream>

template<typename T_Buffer, typename T_Extents>
auto writeImage(uint32_t const currentStep, T_Buffer const& buffer, T_Extents const& extents) -> void
{
    std::stringstream step;
    step << std::setw(6) << std::setfill('0') << currentStep;
    std::string filename("heat_" + step.str() + ".png");
    pngwriter png{static_cast<int>(extents[1]), static_cast<int>(extents[0]), 0, filename.c_str()};
    png.setcompressionlevel(9);

    for(uint32_t y = 0; y < extents[0]; ++y)
    {
        for(uint32_t x = 0; x < extents[1]; ++x)
        {
            auto p = buffer.data()[y * extents[1] + x];
            png.plot(x + 1, extents[0] - y, p, 0., 1. - p);
        }
    }
    png.close();
}
