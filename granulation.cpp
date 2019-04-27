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
#include "granulation.h"
#include "perlin.h"
#include <opencv2/opencv.hpp>

using namespace cv;
void addGranulationEffect(Mat & img)
{
	Mat noise = Mat::zeros(img.size(), CV_32F), tmp;
	double scale = 0.5;
	const int cnt = 4;
	for (int i = 0; i < cnt; ++i)
	{
		getPerlinNoise(tmp, img.size(), scale); 
		noise += tmp * (8.0 * scale);
		scale /= 2.0;
	}
	normalize(noise, noise, 0.0, 1.0, NORM_MINMAX);
	noise *= 0.25;
	noise += 0.875;
	Mat noise3[] = { noise, noise, noise };
	merge(noise3, 3, noise);
	img.convertTo(img, CV_32F, 1.0 / 255.0);
	img.mul(noise);
	img.convertTo(img, CV_8UC3, 255.0);
	imshow("granulation", img);
}
