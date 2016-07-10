//////////////////////////////////////////////////////////////////////////
// Name:	    chars_identify Header
// Version:		1.0
// Date:	    2014-09-25
// Author:	    liuruoze
// Copyright:   liuruoze
// Reference:	Mastering OpenCV with Practical Computer Vision Projects
// Reference:	CSDN Bloger taotao1233
// Desciption:
// Defines CCharsIdentify 
//////////////////////////////////////////////////////////////////////////
#ifndef __CHARS_IDENTIFY_H__
#define __CHARS_IDENTIFY_H__

#ifndef __PREP_H__
#define __PREP_H__

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv/cvaux.h"

#include <objbase.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <math.h> 

using namespace std;
using namespace cv;

#endif
/* endif __PREP_H__ */

/*! \namespace easypr
    Namespace where all the C++ EasyPR functionality resides
*/

	
class CCharsIdentify 
{
public:
	CCharsIdentify();

	//! 字符分割
	string charsIdentify(Mat, bool);

	//! 字符分类
	int classify(Mat, bool);

	//create the accumulation histograms,img is a binary image, t is 水平或垂直
	Mat ProjectedHistogram(Mat img, int t);

	//! 获得字符的特征图
	Mat features(Mat in, int sizeData);

	//! 装载ANN模型
	void LoadModel();

	//! 装载ANN模型
	void LoadModel(string s);

	//! 设置与读取模型路径
	inline void setModelPath(string path){	m_path = path;	}
	inline string getModelPath() const{	 return m_path;	}

private:
	//！使用的ANN模型
	CvANN_MLP ann;

	//! 模型存储路径
	string m_path;

	//! 特征尺寸
	int m_predictSize;

	//! 省份对应map
	map<string, string> m_map;
};

	/* \namespace easypr  */

#endif /* endif __CHARS_IDENTIFY_H__ */