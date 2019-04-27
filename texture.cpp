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
#include "texture.h"
#include <vector>
using namespace std;
void addTextureEffect(Mat& img)
{
	Mat texture = imread("texture.jpg", cv::ImreadModes::IMREAD_GRAYSCALE);
	Mat add(img.size(), CV_8U);

	for (int i = 0; i < add.rows; i++)
		for (int j = 0; j < add.cols; j++) {
			int x = texture.at<uchar>(i % texture.rows, j % texture.cols);
			x = max(x, 230);
			add.at<uchar>(i, j) = min(x, 255);
		}
	imshow("add", add);
	waitKey(0);
	add.convertTo(add, CV_32F, 1.0 / 255.0);
	Mat tmpadd[] = { add, add, add };
	merge(tmpadd, 3, add);
	img.convertTo(img, CV_32F, 1.0 / 255.0);
	img = img.mul(add);
	img.convertTo(img, CV_8UC3, 255.0);
	imshow("textureimg", img);
}