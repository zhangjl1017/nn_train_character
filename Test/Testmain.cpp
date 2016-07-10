//The example of using BPNetwork in OpenCV
//Coded by L. Wei
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "chars.h"
using namespace std;
using namespace cv;

#define HORIZONTAL    1
#define VERTICAL    0
//中国车牌
const char strCharacters[] = {'0','1','2','3','4','5',\
	'6','7','8','9','A','B', 'C', 'D', 'E','F', 'G', 'H', /* 没有I */\
	'J', 'K', 'L', 'M', 'N', /* 没有O */ 'P', 'Q', 'R', 'S', 'T', \
	'U','V', 'W', 'X', 'Y', 'Z'}; 
const int numCharacter = 34; /* 没有I和O,10个数字与24个英文字符之和 */

//以下都是我训练时用到的中文字符数据，并不全面，有些省份没有训练数据所以没有字符
//有些后面加数字2的表示在训练时常看到字符的一种变形，也作为训练数据存储
const string strChinese[] = {"zh_cuan" /* 川 */, "zh_e" /* 鄂 */,  "zh_gan" /* 赣*/, \
	"zh_hei" /* 黑 */, "zh_hu" /* 沪 */,  "zh_ji" /* 冀 */, \
	"zh_jl" /* 吉 */, "zh_jin" /* 津 */, "zh_jing" /* 京 */, "zh_shan" /* 陕 */, \
	"zh_liao" /* 辽 */, "zh_lu" /* 鲁 */, "zh_min" /* 闽 */, "zh_ning" /* 宁 */, \
	"zh_su" /* 苏 */,  "zh_sx" /* 晋 */, "zh_wan" /* 皖 */,\
	 "zh_yu" /* 豫 */, "zh_yue" /* 粤 */, "zh_zhe" /* 浙 */};

const int numChinese = 20;
const int numAll = 54; /* 34+20=54 */
CvANN_MLP  ann;
const int numNeurons = 20;
const int predictSize = 10;
Mat features(Mat in, int sizeData);
int saveTrainData();
Mat ProjectedHistogram(Mat img, int t);
Mat features(Mat in, int sizeData);

int saveTrainData()
{
	cout << "Begin saveTrainData" << endl;
    Mat classes;
    Mat trainingDataf5;
    Mat trainingDataf10;
    Mat trainingDataf15;
    Mat trainingDataf20;

    vector<int> trainingLabels;
	string path = "C:/Users/zjl/Desktop/EasyPR-1.2/EasyPR-1.2/train/data/chars_recognise_ann/chars2/chars2";

    for(int i = 0; i < numCharacter; i++)
    {
		cout << "Character: "<< strCharacters[i] << "\n";
        stringstream ss(stringstream::in | stringstream::out);
        ss << path << "/" << strCharacters[i];

		vector<string> files;
		getFiles(ss.str(), files);

		int size = files.size();
		for (int j = 0; j < size; j++)
		{
			cout << files[j].c_str() << endl;
			Mat img = imread(files[j].c_str(), 0);
            Mat f5=features(img, 5);
            Mat f10=features(img, 10);
            Mat f15=features(img, 15);
            Mat f20=features(img, 20);

            trainingDataf5.push_back(f5);
            trainingDataf10.push_back(f10);
            trainingDataf15.push_back(f15);
            trainingDataf20.push_back(f20);
            trainingLabels.push_back(i);			//每一幅字符图片所对应的字符类别索引下标
		}
    }
   
	path = "C:/Users/zjl/Desktop/EasyPR-1.2/EasyPR-1.2/train/data/chars_recognise_ann/charsChinese/charsChinese";

	for (int i = 0; i < numChinese; i++)
	{
		cout << "Character: "<< strChinese[i] << "\n";
		stringstream ss(stringstream::in | stringstream::out);
        ss << path << "/" << strChinese[i];

		vector<string> files;
		getFiles(ss.str(), files);

		int size = files.size();
		for (int j = 0; j < size; j++)
		{
			cout << files[j].c_str() << endl;
			Mat img = imread(files[j].c_str(), 0);
            Mat f5=features(img, 5);
            Mat f10=features(img, 10);
            Mat f15=features(img, 15);
            Mat f20=features(img, 20);

            trainingDataf5.push_back(f5);
            trainingDataf10.push_back(f10);
            trainingDataf15.push_back(f15);
            trainingDataf20.push_back(f20);
            trainingLabels.push_back(i + numCharacter);			
		}
	}

    trainingDataf5.convertTo(trainingDataf5, CV_32FC1);
    trainingDataf10.convertTo(trainingDataf10, CV_32FC1);
    trainingDataf15.convertTo(trainingDataf15, CV_32FC1);
    trainingDataf20.convertTo(trainingDataf20, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);

    FileStorage fs("C:/Users/zjl/Desktop/EasyPR-1.2/EasyPR-1.2/train/ann_data.xml", FileStorage::WRITE);
    fs << "TrainingDataF5" << trainingDataf5;
    fs << "TrainingDataF10" << trainingDataf10;
    fs << "TrainingDataF15" << trainingDataf15;
    fs << "TrainingDataF20" << trainingDataf20;
    fs << "classes" << classes;
    fs.release();

	cout << "End saveTrainData" << endl;

    return 0;
}
void annTrain(Mat TrainData, Mat classes, int nNeruns)
{
	ann.clear();
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = TrainData.cols;
	layers.at<int>(1) = nNeruns;
	layers.at<int>(2) = numAll;
	ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	//Prepare trainClases
	//Create a mat with n trained data by m classes
	Mat trainClasses;
	trainClasses.create( TrainData.rows, numAll, CV_32FC1 );
	for( int i = 0; i <  trainClasses.rows; i++ )
	{
		for( int k = 0; k < trainClasses.cols; k++ )
		{
			//If class of data i is same than a k class
			if( k == classes.at<int>(i) )
				trainClasses.at<float>(i,k) = 1;
			else
				trainClasses.at<float>(i,k) = 0;
		}
	}
	Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );

	//Learn classifier
	ann.train( TrainData, trainClasses, weights );
}

void saveModel(int _predictsize, int _neurons)
{
	FileStorage fs;
	fs.open("C:/Users/zjl/Desktop/EasyPR-1.2/EasyPR-1.2/train/ann_data.xml", FileStorage::READ);

	Mat TrainingData;
	Mat Classes;

	string training;
	if(1)
	{ 
		stringstream ss(stringstream::in | stringstream::out);
		ss << "TrainingDataF" << _predictsize;
		training = ss.str();
	}

	fs[training] >> TrainingData;
	fs["classes"] >> Classes;

	//train the Ann
	cout << "Begin to saveModelChar predictSize:" << _predictsize 
		<< " neurons:" << _neurons << endl;

	double start = GetTickCount();  
	annTrain(TrainingData, Classes, _neurons);
	double end = GetTickCount();  
	cout << "GetTickCount:" << (end-start)/1000 << endl;  

	cout << "End the saveModelChar" << endl;

	string model_name = "C:/Users/zjl/Desktop/EasyPR-1.2/EasyPR-1.2/train/ann.xml";


	FileStorage fsTo(model_name, cv::FileStorage::WRITE);
	ann.write(*fsTo, "ann");
}

int main()
{
	//saveTrainData();
	//saveModel(10, 20);
	Mat ima=imread("E:\\charsChinese\\charsChinese\\zh_hu\\0-0.jpg");
	cvtColor(ima,ima,CV_BGR2GRAY);
	CCharsIdentify c;
	string out=c.charsIdentify(ima,true);
	cout<<out<<endl;
	getchar();
	/*//Setup the BPNetwork
	CvANN_MLP bp; 
	// Set up BPNetwork's parameters
	CvANN_MLP_TrainParams params;
	params.train_method=CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale=0.1;
	params.bp_moment_scale=0.1;
	//params.train_method=CvANN_MLP_TrainParams::RPROP;
	//params.rp_dw0 = 0.1; 
	//params.rp_dw_plus = 1.2; 
	//params.rp_dw_minus = 0.5;
	//params.rp_dw_min = FLT_EPSILON; 
	//params.rp_dw_max = 50.;

	// Set up training data
	float labels[3][5] = {{0,0,0,0,0},{1,1,1,1,1},{0,0,0,1,1}};
	Mat labelsMat(3, 5, CV_32FC1, labels);

	float trainingData[3][5] = { {1,2,3,4,5},{111,112,113,114,115}, {21,22,23,24,25} };
	Mat trainingDataMat(3, 5, CV_32FC1, trainingData);
	Mat layerSizes=(Mat_<int>(1,5) << 5,2,2,2,5);
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM
	                                           //CvANN_MLP::GAUSSIAN
	                                           //CvANN_MLP::IDENTITY
	bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);


	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	Vec3b green(0,255,0), blue (255,0,0);
	

		imwrite("result.png", image);        // save the image 

		imshow("BP Simple Example", image); // show it to the user
		waitKey(0);
		*/

}

Mat ProjectedHistogram(Mat img, int t)
{
	int sz=(t)?img.rows:img.cols;
	Mat mhist=Mat::zeros(1,sz,CV_32F);

	for(int j=0; j<sz; j++){
		Mat data=(t)?img.row(j):img.col(j);
		mhist.at<float>(j)=countNonZero(data);	//统计这一行或一列中，非零元素的个数，并保存到mhist中
	}

	//Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if(max>0)
		mhist.convertTo(mhist,-1 , 1.0f/max, 0);//用mhist直方图中的最大值，归一化直方图

	return mhist;
}

//! 获得字符的特征图
Mat features(Mat in, int sizeData)
{
	//Histogram features
	Mat vhist=ProjectedHistogram(in, VERTICAL);
	Mat hhist=ProjectedHistogram(in, HORIZONTAL);

	//Low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));
	//Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);

	//Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
	int j=0;
	for(int i=0; i<vhist.cols; i++)
	{
		out.at<float>(j)=vhist.at<float>(i);
		j++;
	}
	for(int i=0; i<hhist.cols; i++)
	{
		out.at<float>(j)=hhist.at<float>(i);
		j++;
	}
	for(int x=0; x<lowData.cols; x++)
	{
		for(int y=0; y<lowData.rows; y++){
			out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
			j++;
		}
	}
	return out;
}

