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
#include "handtremor.h"
#include "perlin.h"
#include <iostream>
#include <omp.h>

void addHandTremorEffect(Mat & img, const Mat & segmentimg, const Mat & edge)
{
	Mat P1 = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat P2 = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat P3 = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat P4 = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat tmp;

	double scale = 2.33;
	const int cnt = 8;

	for (int i = 0; i < cnt; ++i)
	{
		getPerlinNoise(tmp, img.size(), scale); P1 += tmp / scale;
		getPerlinNoise(tmp, img.size(), scale); P2 += tmp / scale;
		getPerlinNoise(tmp, img.size(), scale); P3 += tmp / scale;
		getPerlinNoise(tmp, img.size(), scale); P4 += tmp / scale;
		scale /= 2.0;
	}
	normalize(P1, P1, 6, -4, NORM_MINMAX);
	normalize(P2, P2, 6, -4, NORM_MINMAX);
	normalize(P3, P3, 6, -4, NORM_MINMAX);
	normalize(P4, P4, 6, -4, NORM_MINMAX);

	Mat outputImg = img.clone();
#pragma omp parallel for
	for(int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (edge.at<uchar>(i, j) == 2)
			{
				Point p(j + P2.at<float>(i, j), i + P1.at<float>(i, j));
				if (checkInBoundary(p, img.size()))
					outputImg.at<Vec3b>(i, j) = img.at<Vec3b>(p);
			}
			else if (edge.at<uchar>(i, j) == 3)
			{
				Point pa(j + P2.at<float>(i, j), i + P1.at<float>(i, j));
				Vec3b colora = checkInBoundary(pa, img.size()) && segmentimg.at<Vec3b>(i, j) == segmentimg.at<Vec3b>(pa)
					? img.at<Vec3b>(pa) : img.at<Vec3b>(i, j);
				Point pb(j + P4.at<float>(i, j), i + P3.at<float>(i, j));
				Vec3b colorb = checkInBoundary(pb, img.size()) && segmentimg.at<Vec3b>(i, j) == segmentimg.at<Vec3b>(pb)
					? img.at<Vec3b>(pb) : img.at<Vec3b>(i, j);
				outputImg.at<Vec3b>(i, j) = Vec3b(255, 255, 255) - (Vec3b(255, 255, 255) - colora + Vec3b(255, 255, 255) - colorb);
			}
		}
	img = outputImg;
	imshow("handtremor", img);
}


