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
#include <iostream>
#include <opencv2/opencv.hpp>

#include "uphsv.h"
#include "saliency.h"
#include "abstraction.h"
#include "edge.h"
#include "wet_in_wet.h"
#include "handtremor.h"
#include "edgedarkening.h"
#include "granulation.h"
#include "turbulence.h"
#include "tunecolor.h"
#include "texture.h"
using namespace std;
using namespace cv;

void watercolorization(Mat& img)
{
	upHSV(img);
	Mat saliencyDistanceField;
	getSaliencyDistanceField(img, saliencyDistanceField);
	Mat segmentimg;
	segmentation(img, segmentimg);
	abstraction(img, segmentimg, saliencyDistanceField);
	Mat edge, gradientX, gradientY;
	edgeDetection(img, saliencyDistanceField, edge, gradientX, gradientY);
	addWetEffect(img, segmentimg, edge, gradientX, gradientY);
	addHandTremorEffect(img, segmentimg, edge);
	addEdgeDarkeningEffect(img);
	tunecolor(img);
	addGranulationEffect(img);
	addTurbulenceEffect(img);
	medianBlur(img, img, 3);
	addTextureEffect(img);
	imshow("output", img);
}

int main(int argc, char * argv[])
{
	if (argc != 2)
	{
		cerr << "请输入文件名" << endl;
		exit(EXIT_FAILURE);
	}

	Mat img = imread(argv[1], IMREAD_COLOR);
	if (!img.data)
	{
		cerr << "文件读取错误" << endl;
		exit(EXIT_FAILURE);
	}

	imshow("init_img", img);
	waitKey(0);

	watercolorization(img);

	waitKey(0);
	return 0;
}
	
