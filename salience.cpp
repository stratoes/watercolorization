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
#include "saliency.h"
#include <omp.h>
#include <algorithm>
using namespace std;
using namespace cv;

void openOperation(Mat& src)
{
	Mat& dst = src;
	//开运算，要求相关区域足够大
	erode(src, dst, Mat::ones(7, 7, CV_8U), Point(-1, -1), 3);
	dilate(src, dst, Mat::ones(7, 7, CV_8U), Point(-1, -1), 3);
}

//论文中要求的方法效果更好，但是过于复杂，因此使用其他项目中的实现方法
void getSaliencyBinaryMap(const Mat& img, Mat& saliencyBinaryMap)
{
	auto saliencyBinaryAlgorithm = saliency::StaticSaliencyFineGrained::create();
	//计算显著度
	saliencyBinaryAlgorithm->computeSaliency(img, saliencyBinaryMap);
	//32FC1转到8UC1空间
	saliencyBinaryMap.convertTo(saliencyBinaryMap, CV_8UC1, 255.0);
	//提取相关区域
	threshold(saliencyBinaryMap, saliencyBinaryMap, 255 >> 1, 255, THRESH_BINARY | THRESH_OTSU);

	openOperation(saliencyBinaryMap);

	Mat lut(1, 256, CV_8U);
	lut.at<uchar>(0) = GC_PR_BGD;
	lut.at<uchar>(255) = GC_PR_FGD;
	LUT(saliencyBinaryMap, lut, saliencyBinaryMap);

	//掩码图案初始化
	grabCut(img, saliencyBinaryMap, Rect(), Mat(), Mat(), 1, GC_INIT_WITH_MASK);
	//图像分割
	grabCut(img, saliencyBinaryMap, Rect(), Mat(), Mat(), 4, GC_EVAL);

	Mat lutinv(1, 256, CV_8U);
	lut.at<uchar>(GC_BGD) = lut.at<uchar>(GC_PR_BGD) = 0;
	lut.at<uchar>(GC_FGD) = lut.at<uchar>(GC_PR_FGD) = 255;
	LUT(saliencyBinaryMap, lut, saliencyBinaryMap);

	openOperation(saliencyBinaryMap);

	dilate(saliencyBinaryMap, saliencyBinaryMap, Mat::ones(7, 7, CV_8U), Point(-1, -1), 1);
	imshow("saliencyBinaryMap", saliencyBinaryMap);
	waitKey(0);
}

void getNormalizedDistanceField(const Mat& saliencyBinaryMap, Mat& normalizedDistanceField)
{
	Mat saliencyBinaryMapInv, dis_o, tag;
	bitwise_not(saliencyBinaryMap, saliencyBinaryMapInv);
	distanceTransform(saliencyBinaryMapInv, dis_o, tag, DIST_L2, DIST_MASK_PRECISE, DIST_LABEL_PIXEL);
	double maxidx; minMaxIdx(tag, NULL, &maxidx);
	vector<Point> minPos(int(maxidx) + 1);
	for (int i = 0; i < tag.rows; ++i)
		for (int j = 0; j < tag.cols; ++j)
			if (dis_o.at<float>(i, j) == 0.0)
				minPos[tag.at<int>(i, j)] = Point(j, i);
	normalizedDistanceField = dis_o.clone();
#pragma omp parallel for
	for (int i = 0; i < tag.rows; ++i)
		for (int j = 0; j < tag.cols; ++j)
			if(dis_o.at<float>(i, j) != 0.0)
			{
				Point minPosPoint = minPos[tag.at<int>(i, j)];
				Vec2f dis(j - minPosPoint.x, i - minPosPoint.y);
				float t = FLT_MAX;
				if (dis[0] > 0.0f) t = min(t, (float)(tag.cols - 1 - minPosPoint.x) / dis[0]);
				else if (dis[0] < 0.0f) t = min(t, -(float)(minPosPoint.x) / dis[0]);
				if (dis[1] > 0.0f) t = min(t, (float)(tag.rows - 1 - minPosPoint.y) / dis[1]);
				else if (dis[1] < 0.0f) t = min(t, -(float)(minPosPoint.y) / dis[1]);
				Point2f borderPoint2f = Vec2f(minPosPoint.x, minPosPoint.y) + t * dis;
				Point borderPoint(borderPoint2f.x, borderPoint2f.y);
				normalizedDistanceField.at<float>(i, j) =
					min(1.0f, dis_o.at<float>(i, j) / dis_o.at<float>(borderPoint.y, borderPoint.x));
			}
	imshow("normalized distance field", normalizedDistanceField);
	waitKey(0);
}

void getSaliencyDistanceField(const Mat& img, Mat& saliencyDistanceField)
{
	Mat saliencyBinaryMap, normalizedDistanceField;
	getSaliencyBinaryMap(img, saliencyBinaryMap);
	getNormalizedDistanceField(saliencyBinaryMap, normalizedDistanceField);
	saliencyDistanceField = normalizedDistanceField.clone();
	for (int i = 0; i < 5; ++i) GaussianBlur(saliencyDistanceField, saliencyDistanceField, Size(5, 5), 0, 0);
}