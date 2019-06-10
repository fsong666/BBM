/**
 * Bildbasierte Modellierung SS 2019
 * Prof. Dr.-Ing. Marcus Magnor
 *
 * Betreuer: JP Tauscher (tauscher@cg.cs.tu-bs.de)
 * URL: https://graphics.tu-bs.de/teaching/ss19/bbm
 */

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include "PFMReadWrite.h"
using namespace cv;
using namespace std;
/**
 * Datum: 28.5.2019
 *
 * Übungsblatt: 6
 * Abgabe: 4.6.2019
 */

/**
 * Die Middleburry Stereo Datasets (\url{http://vision.middlebury.edu/stereo/data/}) dienen als Benchmark
 * für Algorithmen zur Schätzung von Tiefenkarten aus rektifizierten Stereo-Bildpaaren. Die Scores können unter
 * \url{http://vision.middlebury.edu/stereo/eval3/} eingesehen werden.
 */
//cam0=[7190.247 0 1938.523; 0 7190.247 913.94; 0 0 1]
//cam1=[7190.247 0 2293.672; 0 7190.247 913.94; 0 0 1]
//Z = baseline * f / (d + doffs)
const float fx = 7190.247 ;
const float fy = 7190.247 ;
const float cx = 1938.523 ;
const float cy = 913.94 ;
const float doffs = 355.149 ;
const float baseline = 174.945 ;
const int radius = 3;

void filter( Mat &src, Mat &dst,  Mat &kernel ){
    int	size = kernel.rows;
    int border = (size - 1)/ 2 ;//das Radius der Kernel
    dst = src.clone();
//    cout<<"kernel:" << kernel << endl;
//    cout<<"dst.type(): " <<dst.type() << endl;

    Mat padded;
    copyMakeBorder(src, padded, border, border, border, border, BORDER_CONSTANT, Scalar::all(0));
    Mat window;
    double max = 0;
    for(int y = 0 ; y < src.rows  ; y++ ){
        for(int x = 0;x < src.cols; x++  ){
            //the maxmum of local region with the radius finden
            padded( Rect( x , y, size , size) ).copyTo(window);
            minMaxLoc(window, NULL, &max);
            dst.at<ushort>(y , x) = ushort( max );
        }

    }
}

void dispToDepth1(cv::Mat &dispMap, cv::Mat &depthMap)
{
    Mat disMapFilter;// = dispMap.clone();
    depthMap.create( dispMap.rows, dispMap.cols, CV_16UC1);

    if (dispMap.type() == CV_16U)
    {
        Mat kernel = Mat::ones( radius, radius, CV_16UC1 );
        filter( dispMap, disMapFilter ,kernel);

        imshow("after maxfilter", disMapFilter);

        int height = disMapFilter.rows;
        int width = disMapFilter.cols;

        ushort* dispData = (ushort*)disMapFilter.data;
        ushort* depthData = (ushort*)depthMap.data;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int id = i*width + j;
                if (!dispData[id])  continue;  //vermeiden 0 to divide
                depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]+ doffs) );
            }
        }
    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}



void getDescriptor(cv::Mat &dispMap, vector<cv::Mat> &dstVector)
{
    int	size = radius;
    int border = (size - 1)/ 2 ;//das Radius der Kernel

    Mat padded;
    copyMakeBorder(dispMap, padded, border, border, border, border, BORDER_CONSTANT, Scalar::all(0));

    for(int y = 0 ; y < dispMap.rows ; y++ ){
        for(int x = 0;x < dispMap.cols; x++  ){
            Mat window;
            //ein Fenster von mehreren Pixeln Breite
            //jede pixel der Disparitaete has die eingene 3*3Matrix from Nachbarpixel als ihr Descriptor
            padded( Rect( x , y, size , size) ).copyTo(window);
            dstVector.push_back(window);
        }

    }
}


void compareDescriptor(Mat &src, vector<cv::Mat> disMapDescriptor , Mat &dst, int size)
{
    int border = (size - 1)/ 2 ;//das Radius der Kernel
    dst = src.clone();
    Mat averDescriptor( src.size(), CV_32F );

    int height = src.rows;
    int width = src.cols;
    int id = 0;
    //get average of each desriptor
    for(int j = 0; j < height; j++){
        for(int k = 0; k < width ; k++){
            id = j * width + k;
            float average = float( sum( disMapDescriptor[id] )[0] / (size*size ) );
            averDescriptor.at<float>(j, k) = round(average);
        }
    }

    Mat padded;
    copyMakeBorder(averDescriptor, padded, border, border, border, border, BORDER_CONSTANT, Scalar::all(0));
    Mat window;
    double max = 0;
    //the maxmum in local region finden
    for(int y = 0 ; y < height ; y++ ){
        for(int x = 0;x < width; x++  ){

            padded( Rect( x , y, size , size) ).copyTo(window);
            minMaxLoc(window, NULL, &max);
            dst.at<ushort>(y , x) = ushort( max );
        }

    }
}

void dispToDepth2(cv::Mat &dispMap, cv::Mat &depthMap)
{
    depthMap.create( dispMap.rows, dispMap.cols, CV_16UC1);

    vector<cv::Mat> disMapDescriptor;
    getDescriptor(dispMap, disMapDescriptor);

    if (dispMap.type() == CV_16U)
    {
        Mat comparedDisMap;
        compareDescriptor(dispMap, disMapDescriptor, comparedDisMap, radius);

        imshow("after Descriptor compared", comparedDisMap);

        int height = depthMap.rows;
        int width = depthMap.cols;

        ushort* dispData = (ushort*)comparedDisMap.data;
        ushort* depthData = (ushort*)depthMap.data;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int id = i*width + j;
                if (!dispData[id])  continue;  //vermeiden 0 to divide
                depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]+ doffs) );
            }
        }
    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}


int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <image_1> " << "<image_2>" << std::endl;
        exit(1);
    }

     Mat img1, img2;

    img1 = imread(argv[1], -1);
    img2 = imread(argv[2], -1);

    if (img1.empty() || img1.empty()) {
        std::cout << "Could not load image file: " << argv[1] << " or " << argv[2] << std::endl;
        exit(0);
    }


//    img1_U8.convertTo(img1, CV_32F, 1 / 255.);
//    img2_U8.convertTo(img2, CV_32F, 1 / 255.);


    /**
     * Aufgabe: Erzeugen einer Tiefenkarte (10 Punkte)
     *
     * Implementiere ein eigenes Verfahren zur Tiefenschätzung auf einem Middleburry Bildpaar Deiner Wahl.
     * Der Suchradius soll dabei vorgegeben werden können. Der Pixel in Bild 1 an der Position $\vec{x}_1$
     * ist beschrieben durch einen Deskriptor $d_1(\vec{x}_1)$. Für jeden Pixel $\vec{x}_2$ innerhalb des
     * Suchradius muss nun der Deskriptor $d_2(\vec{x}_2)$ mit $d_1$ verglichen werden. Verwende als Deskriptor
     * zunächst einfach die Farbe des Pixels. Zeige die erzeugte Tiefenkarte unter Verwendung einer geeigneten
     * Color Map an.
     */
    resize( img1, img1, Size( img1.cols / 4, img1.rows/ 4 ) );
    resize( img2, img2, Size( img2.cols / 4, img2.rows / 4 ) );

    imshow("Input Image 1", img1 );
    imshow("Input Image 2", img1 );
    cout <<"input size: " << img1.size() << endl;
    cout <<"Input1 type: " << img1.type() << endl;
    cout <<"Input2 type: " << img2.type() << endl;
    cout <<"radius: " << radius << endl;

    waitKey(0);

    //input CV_16UC1
    Mat depthImg1, depthImg2;
    dispToDepth1( img1, depthImg1 );
    dispToDepth1( img1, depthImg2 );


    Mat colorDepth1, colorDepth2;
    depthImg1.convertTo(depthImg1, CV_8U);
    depthImg2.convertTo(depthImg2, CV_8U);
//    imshow("output depth Image 1", depthImg1);
//    imshow("output depth Image 2", depthImg2);

    applyColorMap( depthImg1, colorDepth1, COLORMAP_HOT);
    applyColorMap( depthImg2, colorDepth2, COLORMAP_HOT);
    imshow("output depth Image color1", colorDepth1 );
    imshow("output depth Image color2", colorDepth2 );

    waitKey(0);

    /**
     * Aufgabe: Robustere Methoden (20 Punkte)
     *
     * In dieser Aufgabe soll die Tiefenschätzung robuster gegenüber Fehlern
     * gemacht werden.  Hier ist Deine Kreativität gefragt.  Überlege Dir wie die
     * Disparität zuverlässiger bestimmt werden kann und implementiere Deinen
     * Ansatz. Möglich wären zum Beispiel:
     * - bessere Deskriptoren, etwa ein Fenster von mehreren Pixeln Breite
     * - Regularisierung, d.h. benachbarte Pixel sollten ähnliche Tiefenwerte
     *   haben, auch wenn dadurch die Deskriptoren etwas weniger gut passen; dazu
     *   könnte man beispielsweise mit der Lösung der ersten Aufgabe beginnen und
     *   in einem zweiten Schritt die Disparitäten der Nachbarpixel in den
     *   Deskriptor mit einbauen. Das Ganze würde man dann solange wiederholen, bis
     *   sich nichts mehr ändert.
     * - Weitere Inspiration kann beim Stöbern durch die Paper in den Middleburry-Scores
     *   gefunden werden.
     */
    dispToDepth2( img1, depthImg1 );
    dispToDepth2( img1, depthImg2 );


    depthImg1.convertTo(depthImg1, CV_8U);
    depthImg2.convertTo(depthImg2, CV_8U);

    applyColorMap( depthImg1, colorDepth1, COLORMAP_HOT);
    applyColorMap( depthImg2, colorDepth2, COLORMAP_HOT);
    imshow("depth Image color1 bessere Robust ", colorDepth1 );
    imshow("depth Image color2 bessere Robust ", colorDepth2 );

    waitKey(0);
    return 0;

}
