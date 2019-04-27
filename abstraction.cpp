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
#include "abstraction.h"
#include <algorithm>
#include <opencv2/hfs.hpp>
#include <omp.h>
using namespace std;
const double saliencyThreshold = 1e-8;
void segmentation(const Mat & img, Mat & segmentimg)
{
	segmentimg = hfs::HfsSegment::create(img.rows, img.cols, 0.01f, 500, 0.01f, 500, 0.3f, 40, 60)->performSegmentCpu(img, true);
	imshow("segmentimg", segmentimg);
	waitKey(0);
}

void abstraction(Mat & img, const Mat & segmentimg, const Mat & saliencyDistanceField)
{
	Mat outputImg(img.size(), CV_8UC3);
#pragma omp parallel for
	for(int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3f sum(0.0, 0.0, 0.0);
			int cnt = 0, k = 3;
			bool saliency = true;
			float d = saliencyDistanceField.at<float>(i, j);
			if (d > saliencyThreshold)
			{
				saliency = false;
				k = min(max(int(22 * (d + 0.3f)), 16), 25) >> 1;
			}
			for (int u = max(0, i - k); u <= min(img.rows - 1, i + k); ++u)
				for (int v = max(0, j - k); v <= min(img.cols - 1, j + k); ++v)
				{
					if (segmentimg.at<Vec3b>(i, j) == segmentimg.at<Vec3b>(u, v))
						sum += img.at<Vec3b>(u, v);
					else if (!saliency && fabs(saliencyDistanceField.at< float >(i, j) - saliencyDistanceField.at< float >(u, v)) < d * 0.01f)
						sum += img.at<Vec3b>(u, v);
					else
						sum += img.at<Vec3b>(i, j);
					++cnt;
				}
			outputImg.at<Vec3b>(i, j) = sum / cnt;
		}

	img = outputImg;
	addWeighted(img, 0.4, segmentimg, 0.6, 0, img);
	
	for (int i = 0; i < 5; ++i) //根据论文，做连续的开闭运算
	{
		morphologyEx(img, img, MORPH_OPEN, Mat());
		morphologyEx(img, img, MORPH_CLOSE, Mat());
	}
	imshow("abstraction", img);
	waitKey(0);
}