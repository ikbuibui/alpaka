/* Copyright 2013-2023 Heiko Burau, Rene Widera, Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pngwriter.h>

#include <cstdint>
#include <iomanip>
#include <sstream>

struct PngCreator
{
    template<typename T_Buffer, typename T_Extents>
    void operator()(uint32_t currentStep, T_Buffer data, T_Extents extents)
    {
        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << currentStep;
        std::string filename("heat_" + step.str() + ".png");
        pngwriter png(extents[1], extents[0], 0, filename.c_str());
        png.setcompressionlevel(9);

        for(int y = 0; y < extents[0]; ++y)
        {
            for(int x = 0; x < extents[1]; ++x)
            {
                float p = data[y * extents[1] + x];
                png.plot(x + 1, extents[0] - y, p, 0., 1. - p);
            }
        }
        png.close();
    }
};
