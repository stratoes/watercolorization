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
#include "edge.h"
#include <algorithm>
#include <omp.h>
using namespace std;
void edgeDetection(const Mat & img, const Mat & saliencyDistanceField, Mat & edge, Mat & gradientX, Mat & gradientY)
{
	Mat grayImg, hsv;
	Mat wet_in_wet(img.size(), CV_8UC1, Scalar(0));
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	cvtColor(img, hsv, COLOR_BGR2HSV);
	Scharr(grayImg, gradientX, CV_16S, 1, 0);
	Scharr(grayImg, gradientY, CV_16S, 0, 1);
	Mat absGradientX, absGradientY;
	convertScaleAbs(gradientX, absGradientX);
	convertScaleAbs(gradientY, absGradientY);
	addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, edge);
	threshold(edge, edge, 250, 255, THRESH_BINARY | THRESH_OTSU);
	erode(edge, edge, Mat());
#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			if (edge.at<uchar>(i, j) == 255)
			{
				short dx = gradientX.at<short>(i, j);
				short dy = gradientY.at<short>(i, j);
				int mod = sqrt(sqr(dx) + sqr(dy));
				Point brightPoint(j + dx * 3 / mod, i + dy * 3 / mod);
				Point darkPoint(j - dx * 3 / mod, i - dy * 3 / mod);
				if (!checkInBoundary(brightPoint, img.size()) || !checkInBoundary(darkPoint, img.size())) continue;
				if ((saliencyDistanceField.at<float>(i, j) < 0.3 && calcAngle(hsv.at<Vec3b>(brightPoint)[0], hsv.at<Vec3b>(darkPoint)[0]) < 5) ||
					(saliencyDistanceField.at<float>(i, j) >= 0.3 && calcAngle(hsv.at<Vec3b>(brightPoint)[0], hsv.at<Vec3b>(darkPoint)[0]) < 20))
					edge.at<uchar>(i, j) = 1, wet_in_wet.at<uchar>(i, j) = 255;
				//hand tremor 2 similar hues distorted
				else if (abs(int(hsv.at<Vec3b>(brightPoint)[0]) - int(hsv.at<Vec3b>(darkPoint)[0])) < 45)
				{
					for (int u = i - 3; u <= i + 3; ++u)
						for (int v = j - 3; v <= j + 3; ++v)
							if (checkInBoundary(Point(v, u), img.size()) && edge.at<uchar>(u, v) != 1)
								edge.at<uchar>(u, v) = 2;
				}
				else
				{
					for (int u = i - 3; u <= i + 3; ++u)
						for (int v = j - 3; v <= j + 3; ++v)
							if (checkInBoundary(Point(v, u), img.size()) && edge.at<uchar>(u, v) != 1 && edge.at<uchar>(u, v) != 2)
								edge.at<uchar>(u, v) = 3;
				}
			}
			else
				edge.at<uchar>(i, j) = 0;
}