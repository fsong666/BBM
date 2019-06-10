/**
 * Bildbasierte Modellierung SS 2019
 * Prof. Dr.-Ing. Marcus Magnor
 *
 * Betreuer: JP Tauscher (tauscher@cg.cs.tu-bs.de)
 * URL: https://graphics.tu-bs.de/teaching/ss19/bbm
 */

#include <iostream>
#include <queue>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;
/**
 * Datum: 24.4.2019
 *
 * Übungsblatt: 3
 * Abgabe: 30.4.2019
 */

/**
 * Aufgabe: Median-Filter (10 Punkte)
 * Der Median-Filter ist ein einfacher nichtlinearer Filter, der sich
 * gut eignet, um bestimmte Arten von Bildrauschen zuentfernen.
 * - Implementiere einen Median-Filter, ohne \code{medianBlur} zu verwenden.
 *
 */

/* TODO */
void SaltAndPepper(cv::Mat &image, int n)
{
    for(int k=0;k<n;k++)
    {
        int i=rand()%image.cols;//random value 0-imge.cols;
        int j=rand()%image.rows;


        image.at<cv::Vec3b>(i, j)[0] = 255;
        image.at<cv::Vec3b>(i, j)[1] = 255;
        image.at<cv::Vec3b>(i, j)[2] = 255;
    }
    for(int k=0;k<n;k++)
    {
        int i=rand()%image.cols;
        int j=rand()%image.rows;

        image.at<cv::Vec3b>(i, j)[0] = 0;
        image.at<cv::Vec3b>(i, j)[1] = 0;
        image.at<cv::Vec3b>(i, j)[2] = 0;
    }
}

void myMedianFilter( Mat &src, Mat &dst, int ksize ){
    int	size = ksize;
    int border = (size - 1)/ 2 ;//das Radius der Kernel
    dst = src.clone();

    Mat padded;
    copyMakeBorder(src, padded, border, border, border, border, BORDER_CONSTANT, Scalar::all(0));
    Mat window;

    for(int y = 0 ; y < padded.rows - (size - 1 ) ; y++ ){
        for(int x = 0;x < padded.cols- (size - 1 ); x++  ){

            padded( Rect( x , y, size , size) ).copyTo(window);
            window.reshape(1, 1 );
            cv::sort(window, window, SORT_EVERY_ROW | SORT_ASCENDING );
            dst.at<uchar>(y, x) = window.at<uchar>(window.rows*window.rows /2  );
        }

    }
}


Mat myHough(  Mat &src, vector<Vec2f> &lines, double rho, double theta,int threshold){
    int width = src.cols;
    int height = src.rows;
    int numangle = round( CV_PI / theta ) *2;//180*2
    int amplitude = round(sqrt( width*width + height*height ));
    int numrho = amplitude*2 / rho;
    int r =0, n=0;

    //sky:3846 quard:1510 fence:2560
    //int accum[360][2560];
    vector< vector<int> > accum( numangle, vector<int>( numrho ) );
    for( n = 0; n < numangle; n++){
        for(r = 0; r < numrho; r++ ){
            accum[n][r] = 0;
        }
    }

    unsigned int total = 0;
    bool flag = true;
    Mat houghRaum2(amplitude*2, 360, CV_8UC1, Scalar(0) );
    for( int j = 0; j < height; j++ ){
        for(int k = 0 ; k < width; k++){

            if( src.at<uchar>(j,k) > 250){
                if(total > 255){
                    total = 0;
                    flag = false;
                }

                for( n = 0; n < numangle; n++ ){
                    r = round( k *cos(n) + j * sin(n) );
                    r += amplitude;

                    if( accum[n][r]< INT_MAX && accum[n][r] > INT_MIN  ){
                        accum[n][r]++ ;
                        if(flag){
                            houghRaum2.at<uchar>(r, n ) = total;
                        }
                    }else{
                        accum[n][r] = 0;
                    }

                }

                total++;
            }
        }
    }

    for( n = 0; n < numangle; n++){
        for(r = 0; r < numrho; r++ ){
            if(n-1 > -1 && n+1 < numangle){
                if( accum[n][r] > threshold&&
                    accum[n][r] > accum[n-1][r] && accum[n][r] >= accum[n+1][r]){
                    Vec2f count = Vec2f((float)(r - amplitude) ,(float)( n ) );
                    lines.push_back( count );

                }
            }

        }

    }
    return houghRaum2;

}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image>" << std::endl;
        exit(1);
    }

    Mat img, imgCopy, imgSalt, imgSaltGray, dst;
    img = imread( argv[1], -1 );
    if( img.empty() ){
        cout << " error: cann't open img! " << endl;
        return -1;
    }

    imshow("img", img );
    img.copyTo( imgSalt );
    img.copyTo( imgCopy);
    waitKey(0);

    SaltAndPepper(imgSalt, 6000);
    imshow("imgSalt", imgSalt);

    cvtColor( imgSalt, imgSaltGray, CV_BGR2GRAY );
    imshow("imgSaltGray", imgSaltGray );
    waitKey(0);

    /**
     * - Wende den Median-Filter auf ein Graustufenbild an.
     */
    const int KSIZE = 3;
    myMedianFilter( imgSaltGray , dst, KSIZE);
    imshow("imgSaltGray after myMedianFilter", dst);
    waitKey(0);
/* TODO */


    /**
     * - Wende den Median-Filter auf die einzelnen Kanäle eines Farbbilds an
     */
    vector<Mat> bgr, dstBgr;
    split( imgSalt, bgr );
    imshow("imgSalt[0]", bgr[0]);
    myMedianFilter( bgr[0] , dst, KSIZE);
    imshow("imgSalt[0] after myMedianFilter", dst );
    dstBgr.push_back( dst );
    myMedianFilter( bgr[1] , dst, KSIZE);
    dstBgr.push_back( dst );
    myMedianFilter( bgr[2] , dst, KSIZE);
    dstBgr.push_back( dst );
    waitKey(0);

    merge( dstBgr, dst );
    imshow("dst after myMedianFilter", dst );
    waitKey(0);

/* TODO */


    /**
     * - Wie kann man ungewollte Farbverschiebungen vermeiden?
     * - Für welche Arten von Rauschen eignet sich der Median-Filter gut, für welche nicht?
     */
    //durch bilateral filter nach center Graustufenbild im local Mask
    //Fuer Gauss-weiss Rauschen und SaltRauschen

    /**
     * Aufgabe: Hough-Transformation (10 Punkte)
     *
     * Die Hough-Transformation kann für das Finden von Linien in Bildern verwendet werden.
     *
     * In dieser Aufgabe sollst du die Hough-Transformation implementieren ohne die Funktionen \code{HoughLines}
     * oder \code{HoughLinesP} zu verwenden.
     */


    /**
    * - Erzeuge ein Kantenbild. Verwende dazu einen Filter deiner Wahl. Begründe die Wahl des Kantendetektors.
    */
    Mat midImage,dstImage;
    Canny(imgCopy, midImage, 50, 200, 3);
    cvtColor(midImage,dstImage, CV_GRAY2BGR);

/* TODO */

    /**
    * - Transformiere das Kantenbild in den Hough-Raum und zeige diesen in einer geeigneten Color Map an.
    */
    int amplitude = round(sqrt( midImage.cols*midImage.cols + midImage.rows*midImage.rows ));
    Mat houghRaum(amplitude*2, 360, CV_8UC1, Scalar(0) );

    vector<Vec2f> lines;
    Mat houghRaumcopy =  myHough(midImage, lines, 1, CV_PI/180,400);
    cout<< "lines.size(): "<< lines.size() << endl;

    imshow("houghRaum for 255 points",houghRaumcopy );
    Mat colorMap;
    applyColorMap(houghRaumcopy, colorMap, 	COLORMAP_HOT);
    imshow("houghRaum-color",colorMap );

    for( size_t i = 0; i < lines.size(); i++ ){
        int r = lines[i][0] + amplitude , ang = (int)lines[i][1];
        houghRaum.at<uchar>( r, ang ) = 255;
    }
    imshow("houghRaum of gefunde lines",houghRaum );

    waitKey(0);

/* TODO */

    /**
    * - Finde die markantesten Linien und zeichne diese in das Originalbild ein.
    */

    for( size_t i = 0; i < lines.size(); i++ ){

        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( dstImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
        line( imgCopy, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
    }

    imshow("myHoughLines in CannyImg", dstImage);
    imshow("myHoughLines in src",imgCopy);
    waitKey(0);

/* TODO */

    return 0;

}
