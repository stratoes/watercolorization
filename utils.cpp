/*
 * watercolorization, based on project https://github.com/devin6011/ICGproject-watercolorization
 * Copyright (C) 2019 devin6011
 * Copyright (C) 2019 stratoes
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "utils.h"
#include <cassert>
int sqr(const int & x) { return x * x; }
int calcAngle(const int & a, const int & b) { assert(a >= 0 && a <= 180); assert(b >= 0 && b <= 180); return min(abs(a - b), 180 - abs(a - b)); }
bool checkInBoundary(const Point & p, const Size & imgsize)
{
	return 0 <= p.x && p.x < imgsize.width && 0 <= p.y && p.y < imgsize.height;
}