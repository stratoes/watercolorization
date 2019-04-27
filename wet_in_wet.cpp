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
#include "wet_in_wet.h"
#include <omp.h>
using namespace std;
const double maxDis = 3.0f;
const int kLim = 40;
void addWetEffect(Mat & img, const Mat & segment, const Mat & edge, const Mat & gradientX, const Mat & gradientY)
{
	RNG rng;
#pragma omp parallel for
	for(int i = 0; i < img.rows; ++i)
		for(int j = 0; j < img.cols; ++j)
			if (edge.at<uchar>(i, j) == 1)
			{
				short dx = gradientX.at<short>(i, j);
				short dy = gradientY.at<short>(i, j);
				int mod = sqrt(sqr(dx) + sqr(dy));
				Point brightPoint(j + dx * 3 / mod, i + dy * 3 / mod);
				Point darkPoint(j - dx * 3 / mod, i - dy * 3 / mod);
				Vec3b darkColor = img.at<Vec3b>(darkPoint);
				float scatterDis = rng.uniform(0.0, maxDis);
				Point dstPoint(j + dx * scatterDis / mod, i + dy * scatterDis / mod);
				if (!checkInBoundary(dstPoint, img.size())) continue;
				img.at<Vec3b>(dstPoint) = darkColor;
			}
	imshow("wet-in-wet-scatter", img);
	Mat kernel = Mat::zeros(Size(15, 15), CV_32F);
	for (int i = 0; i < 15; ++i)
	{
		kernel.at<float>(6, i) = (i < 8 ? (i + 1) : (15 - i)) / 32.0;
		kernel.at<float>(7, i) = (i < 8 ? (i + 1) : (15 - i)) / 8.0;
		kernel.at<float>(8, i) = (i < 8 ? (i + 1) : (15 - i)) / 32.0;
	}
	Mat outputImg = img.clone();
	Mat vis = Mat::zeros(img.rows, img.cols, CV_8U);
#pragma omp parallel for
	for(int i = 0; i < img.rows; ++i)
		for(int j = 0; j < img.cols; ++j)
			if (edge.at<uchar>(i, j) == 1)
			{
				short dx = gradientX.at<short>(i, j);
				short dy = gradientY.at<short>(i, j);
				float angle = fastAtan2(dy, dx);
				int mod = sqrt(sqr(dx) + sqr(dy));
				for (int k = 0; k < kLim; ++k)
				{
					Point dstPoint(j + dx * k / mod, i + dy * k / mod);
					if (!checkInBoundary(dstPoint, img.size())) continue;
					if (vis.at<uchar>(dstPoint) == 1) continue;
					vis.at<uchar>(dstPoint) = 1;
					Vec3f sum(0.0, 0.0, 0.0);
					float div = 0.0f;
					Mat rotationMat = getRotationMatrix2D(Point(7, 7), angle, 1.0);
					warpAffine(kernel, kernel, rotationMat, kernel.size());
					for (int u = max(0, dstPoint.y - 7); u <= min(img.rows - 1, dstPoint.y + 7); ++u)
						for (int v = max(0, dstPoint.x - 7); v <= min(img.cols - 1, dstPoint.x + 7); ++v)
						{
							div += kernel.at<float>(u - (dstPoint.y - 7), v - (dstPoint.x - 7));
							sum += img.at<Vec3b>(u, v) * kernel.at<float>(u - (dstPoint.y - 7), v - (dstPoint.x - 7));
						}
					outputImg.at<Vec3b>(dstPoint) = sum / div;
				}
			}
	img = outputImg;
	imshow("wet-in-wet", img);
	waitKey(0);
}