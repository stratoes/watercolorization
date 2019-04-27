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
#include "tunecolor.h"
void tunecolor(Mat & img)
{
	const double alpha = 0.9, beta = 15;
#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			for (int k = 0; k < 3; ++k)
			{
				int cur = (uchar)img.at<Vec3b>(i, j)[k] * alpha + beta;
				if (cur > 255) cur = 255;
				img.at<Vec3b>(i, j)[k] = cur;
			}

	imshow("tunecolor", img);
	waitKey(0);
}