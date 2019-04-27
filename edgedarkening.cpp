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
#include "edgedarkening.h"
#include <opencv2/opencv.hpp>
using namespace cv;

void addEdgeDarkeningEffect(Mat & img)
{
	Mat grayImg, gradientX, gradientY, edge;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	Scharr(grayImg, gradientX, CV_16S, 1, 0);
	Scharr(grayImg, gradientY, CV_16S, 0, 1);
	Mat absGradientX, absGradientY;
	convertScaleAbs(gradientX, absGradientX);
	convertScaleAbs(gradientY, absGradientY);
	addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, edge);
	threshold(edge, edge, 55, 255, THRESH_BINARY | THRESH_OTSU);
	Mat blurredEdge;
	blur(edge, blurredEdge, Size(3, 3));
	dilate(blurredEdge, blurredEdge, Mat::ones(3, 3, CV_8U), Point(-1, -1), 2);
	blur(blurredEdge, blurredEdge, Size(5, 5));
	addWeighted(edge, 0.3, blurredEdge, 0.3, 0, edge);
	GaussianBlur(edge, edge, Size(3, 3), 0, 0);
	edge.convertTo(edge, CV_32F, 1.0 / 255.0);
	edge = 1.3 - 0.5 * edge;
	Mat tmpedge[] = { edge, edge, edge };
	merge(tmpedge, 3, edge);
	img.convertTo(img, CV_32F, 1.0 / 255.0);
	img = img.mul(edge);
	img.convertTo(img, CV_8UC3, 255.0);
	imshow("edgedarkening", img);
}
