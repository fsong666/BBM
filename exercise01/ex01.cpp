/**
 * Bildbasierte Modellierung SS 2019
 * Prof. Dr.-Ing. Marcus Magnor
 *
 * Betreuer: JP Tauscher (tauscher@cg.cs.tu-bs.de)
 * URL: https://graphics.tu-bs.de/teaching/ss18/bbm
 */

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
/**
 * Datum: 10.4.2019
 *
 * Übungsblatt: 1
 * Abgabe: 16.4.2018 
 *
 * Ziel der ersten Übung ist das Vertrautwerden mit der OpenCV-Bibliothek.
 * Diese stellt nicht nur die Grundlagen zum Arbeiten mit Bildern zur Verfügung,
 * sondern auch viele weitere in der Bilderverarbeitung häufig verwendete Funktionen.
 * Erhältlich ist sie als Open-Source-Projekt unter:
 * \begin{center}\url{http://opencv.org/}\end{center}
 * Dokumentation findet sich beispielsweise im Buch \emph{Learning OpenCV} von Gary Bradski und Adrian Kaehler oder unter:
 * \begin{center}\url{http://http://docs.opencv.org/2.4.9/}\end{center}
 */

void inforShow( Mat &img , int &rows, int &cols){
    rows = img.rows;
    cols = img.cols;
    cout << "die Höhe:" << rows<< endl;
    cout << "die Breite:" << cols<< endl;
    cout << "die Farbkanäle:" << img.channels()<< endl;
}

void channelShow( Mat &img,vector<Mat> &planes  ){
    split( img, planes );
    Mat dst( planes[0].rows, planes[0].cols*3,CV_8UC1);
    for (int i = 0 ; i < 3; i++){
        planes[i].copyTo( dst( Rect( planes[0].cols*i, 0 , planes[0].cols, planes[0].rows  ) ) );
    }
    imshow( "output channels", dst);

}

void rechteckShow( Mat &img ){
    Rect select;
    select.x = img.cols/2 - 5;
    select.y = img.rows/2 - 5;
    select.width =10;
    select.height =10;
    rectangle( img, select, cvScalar(0, 0, 255), 3, 4, 0 );
    imshow( "rechteckShow", img );


}

Mat correction(Mat &src, Mat &dst, double k1, double k2){

    const int rows = 240;
    const int cols = 320;
    int xc = cols / 2;
    int yc = rows / 2;
    double r[rows][cols] ;
    double Lr[rows][cols] ;
    int x[rows][cols] ;
    int y[rows][cols] ;

    for( int j = 0; j < rows; j++ ){
        for( int k = 0 ; k < cols; k++ ){
              r[j][k] = double(sqrt( (k - xc)*(k - xc)+ (j - yc)*(j - yc)) );
              Lr[j][k] = 1 + k1*r[j][k] + k2*r[j][k]*r[j][k];
              x[j][k] = xc + Lr[j][k]*( k - xc  );
              y[j][k] = yc + Lr[j][k]*( j - yc  );

        }
    }

    Mat dst2(rows, cols, CV_8UC1, Scalar(0));
    dst2.copyTo(dst);
    for( int j = 0; j < rows ; j++ ){
        for(int k = 0; k < cols ; k++ ){
            if( x[j][k] >= 0 && x[j][k] < cols
                && y[j][k]>= 0 && y[j][k] < rows ){
                dst.at<uchar>(y[j][k] , x[j][k]) = src.at<uchar>(j,k);
            }

        }
    }
    imshow( "planes[0]:" ,src );
    imshow( "dst:" , dst);
    return dst;
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: main <image-file-name> <output-file-name> <kappa1> <kappa2>" << std::endl;
        exit(1);
    }

    /**
     * Aufgabe: OpenCV starten (10 Punkte)
     *
     * Erweitere die gegebene Programmgrundstruktur so, dass
     * - ein Bild geladen werden kann.
     */

     Mat img;
     img = imread( argv[1]);
     if( img.empty() ){
         cout << " error: cann't open img! " << endl;
         return -1;
     }
     int cols;
     int rows;

/* TODO */

    /**
     * - die Höhe, Breite, Anzahl der Farbkanäle dieses Bildes ausgegeben wird.
     */
    inforShow( img, rows, cols );
/* TODO */

    /**
     * - dieses Bild in einem \code{namedWindow} angezeigt wird, bis eine Tastatureingabe erfolgt.
     */
    namedWindow( "inputimg" );
    imshow( "inputimg", img );
    waitKey(0);

/* TODO */

    /**
     * - die drei Farbkanäle des Bildes nebeneinander angezeigt werden.
     */
    vector<Mat> planes;
    channelShow( img, planes );

/* TODO */

    /**
     * - das Bild zusammen mit einem roten $10 \times 10$ Rechteck um die Bildmitte angezeigt wird.
     */
    rechteckShow( img );


    cvWaitKey(0);
/* TODO */
    /**
     * Aufgabe: Bilder entzerren (10 Punkte)
     *
     * Das Bild \code{distorted.png}, wurde mit einer Weitwinkelkamera aufgenommen und zeigt starke radiale Verzerrung.
     * Aus der Vorlesung ist bekannt, dass die radiale Verzerrung oft durch
     * $$ x = x_c + L(r) (x_d-x_c) \quad y = y_c + L(r) (y_d-y_c) $$
     * ausgedrückt wird, wobei $(x, y)$ die idealen Koordinaten sind, $({x_d}, {y_d})$
     * die verzerrten Koordinaten sind und $L({r})$ eine Funktion ist, die nur von der
     * Entfernung ${r}=\sqrt{(x-x_c)^2 + ({y}-y_c)^2}$ zum Verzerrungszentrum
     * $(x_c,y_c)$ abhängt.
     * Die Funktion $L(r)$ kann durch ihre Taylorentwicklung $L(r) = 1+ \kappa_1 r
     * + \kappa_2 r^2 + \kappa_3 r^3 + \cdots$ beschrieben werden.
     * Verschiedene Möglichkeiten, die Parameter zu bestimmen, sind denkbar und
     * werden beispielsweise in \emph{Multiple View Geometry} von Hartley und
     * Zisserman beschrieben, sollen hier aber nicht zur Anwendung kommen.
     *
     * Erweitere die gegebene Programmgrundstruktur so, dass
     * - die Funktion $L$ mit Taylorentwicklung 2. Ordnung approximiert wird, wobei das Verzerrungszentrum der Bildmitte entspricht.
     */
    double k1 = atof(argv[3]);
    double k2 = atof(argv[4]);

    vector<Mat> outimg(3);
    correction( planes[0], outimg[0] ,k1 , k2);
    correction( planes[1], outimg[1] ,k1 , k2);
    correction( planes[2], outimg[2],k1 , k2);

    Mat out(rows, cols, CV_8UC3, Scalar(0,0,0));
    merge( outimg, out );
    imshow( "out:" , out);

    waitKey( 0 );


/* TODO */

    /**
     * - das entzerrte Bild in einer Datei gespeichert wird.
     */
    imwrite( argv[2], out );

/* TODO */

    /**
     * Was passiert, wenn die Größe der Parameter, ihr Vorzeichen etc. verändert wird?
     * Ein Startwert kann z.B. $\kappa_1 = 0.001$, $\kappa_2 = 0.000005$ sein.
     */
    exit(1);
}
