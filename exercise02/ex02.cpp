/**
	* Bildbasierte Modellierung SS 2019
 * Prof. Dr.-Ing. Marcus Magnor
 *
 * Betreuer: JP Tauscher (tauscher@cg.cs.tu-bs.de)
 * URL: https://graphics.tu-bs.de/teaching/ss19/bbm
 */

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;
/**
 * Datum: 17.4.2019
 *
 * Übungsblatt: 2
 * Abgabe: 23.4.2019
 */
void filter( Mat &src, Mat &dst,  Mat &kernel ){
    int	size = kernel.rows;
    int border = (size - 1)/ 2 ;//das Radius der Kernel
    dst = src.clone();
    cout<<"kernel:" << kernel << endl;
    cout<<"dst.type(): " <<dst.type() << endl;

    Scalar sum1 = 0.0;
    Mat padded;
    copyMakeBorder(src, padded, border, border, border, border, BORDER_CONSTANT, Scalar::all(0));
    Mat window;

    for(int y = 0 ; y < padded.rows - (size - 1 ) ; y++ ){
        for(int x = 0;x < padded.cols- (size - 1 ); x++  ){

            padded( Rect( x , y, size , size) ).copyTo(window);
            multiply(window, kernel, window, 1, -1  );
            sum1 = sum( window );
            dst.at<float>(y , x) = sum1[0];
        }

    }
}

void myHarrisCorner( Mat& img, Mat&dst, int ksize, double k){
    int SIZE = ksize;
    Mat dstX, dstY;
    Sobel(img, dstX, CV_32F, 1, 0 , SIZE , 1   );
    Sobel(img, dstY, CV_32F, 0, 1 , SIZE , 1   );

    //Ix2 = dstX^2; Iy2= dstY^2; Ixy=dstX*dstY
    Mat Ix2, Iy2, Ixy;
    multiply(dstX, dstX, Ix2, 1, -1  );
    multiply(dstY, dstY, Iy2, 1, -1  );
    multiply(dstX, dstY, Ixy, 1, -1  );

    vector<Mat> M1;
    Mat dstM;
    M1.push_back( Ix2 );
    M1.push_back( Ixy );
    M1.push_back( Iy2 );
    merge( M1, dstM);

    Mat M;
//    Size s(3,3);
    GaussianBlur(dstM, M, Size(3,3) , 1.0, 1.0);

    vector<Mat> dstM2;
    split(M, dstM2 );

    //detM=g(Ix2)*g(Iy2) - g(Ixy)^2
    Mat detM,traceM;
    detM = dstM2[0].mul(dstM2[2]) - dstM2[1].mul(dstM2[1]);
    //traceM = g(Ix2) + g(Iy2)
    traceM = dstM2[0] + dstM2[2];

    Mat R;
    //R = detM - k*traceM^2
    R = detM - k*traceM.mul(traceM);
    cout<<"R.type()" << R.type()<< endl;
    R.copyTo(dst);

}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image>" << std::endl;
        exit(1);
    }

    /**
     * Aufgabe: 2D-Operationen auf Bildern (5 Punkte)
     *
     * In der Vorlesung wurde gezeigt, das große Bereiche natürlicher Bilder
     * homogen sind, die meisten Informationen über ein Bild jedoch in den
     * Kanten und Diskontinuitäten zu finden sind. In disem Übungsblatt soll
     * es darum gehen, Kanten und herausragende Punkte in Bildern zu finden.
     *
     * Häufige Grundlage der Verarbeitung von Bildern ist das Anwenden von
     * Filtern.  Es entspricht der Faltung (engl. \emph{convolution}) des
     * Bildes mit einem Filterkern. Filterkerne werden zu sehr verschiedenen
     * Zwecken eingesetzt.
     *
     * - Skizziere (auf Papier) eine eindimensionale Gaußfunktion mit
     *   Mittelwert $\mu$ und Varianz $\sigma^2$.  Was ist die
     *   Fourier-Transformation einer Gaußfunktion?
     * - Lade ein Testbild und wandle es anschließend in ein Grauwertbild mit
     *   float-Werten in $[0, 1]$ um.
     */

/* TODO */

    /**
     * - Falte ein verrauschtes Testbild mit Gaußfunktionen verschiedener
     *   Varianzen. Was passiert? Welchen Einfluss hat die Kernelgröße?
     */
    Mat img,dst, dst2,src;
    img = imread( argv[1], IMREAD_GRAYSCALE);
 //   img.copyTo(src);
    if( img.empty() ){
        cout << " error: cann't open img! " << endl;
        return -1;
    }


    //float-Werten in $[0, 1]$
    Mat img0_1;
    img.convertTo(img0_1, CV_32FC1,1 / 255.0);
    imshow("img", img0_1);
    cout <<"img:" << img0_1.type()<<" channel:" << img0_1.channels()<< endl;
    cout<<"img.at<>:  " <<img0_1.at<float>(2,3)<< endl;

//    cornerHarris(img0_1, dst, 3, 3, 0.01);
//    threshold(dst, dst, 0.0001, 255, THRESH_BINARY);
//    imshow("cornerHarris", dst);
//    waitKey(0);

    //gaussfilter
    //je Kernelgröße desto bessere Glaettung
    GaussianBlur(img0_1, dst, Size(3 , 3), 0, 0 );
    GaussianBlur(img0_1, dst2, Size(7 , 7), 0, 0 );
    dst.copyTo(src);
    imshow("GaussianBlur-(3,3)", dst);
    imshow("GaussianBlur-(7,7)", dst2);
    waitKey(0);
/* TODO */

    /**
     * - Betrachte die Differenzen zweier gaußgefilterter Bilder (evt.
     *   skalieren). Wie sind die Nulldurchgänge zu interpretieren?
     */
    Laplacian( img0_1, dst, img0_1.depth(), 3 );
   // convertScaleAbs( dst, dst );
    imshow("Laplacian:", dst);
    waitKey(0);
/* TODO */

    /**
     * Aufgabe: Diskrete Ableitungen (5 Punkte)
     *
     * Mathematisch sind Ableitungen nur für stetige Funktionen definiert.  Da
     * ein Bild nur pixelweise, d.h. an diskreten Stellen, Informationen
     * liefert, kann man Ableitungen von Bildern nicht direkt bestimmen.  Eine
     * naheliegene Approximation der Ableitung ist der Differenzenquotient.
     * Sei $f:\Omega \subset \mathbb{R} \to \mathbb{R}$ eine Funktion.  Dann
     * ist der Differenzenquotient $D_h(x) = \frac{f(x+h) - f(x)}{h}$ eine
     * Approximation an $f'(x)$ für hinreichend kleines h. Für
     * differenzierbare Funktionen liefert allerdings die zentrale Differenz
     * \begin{equation}
     * D(x) = \frac{f(x+h) - f(x-h)}{2h}
     * \end{equation}
     * eine deutlich bessere Approximation (siehe auch \emph{Numerical Recipes
     * in C} von Press et al., Chapter 5).
     *
     * - Bestimme je einen diskreten Faltungskern, der die zentrale Differenz
     *   pro Bildachse approximiert.
     */
    Mat kernelX =(Mat_<float>(3,3) << -1.0/3, 0, 1.0/3,
                                     -1.0/3, 0, 1.0/3,
                                     -1.0/3, 0, 1.0/3 );
    Mat kernelY =(Mat_<float>(3,3) << -1.0/3, -1.0/3, -1.0/3,
                                      0,        0,       0,
                                      1.0/3,  1.0/3,  1.0/3 );

/* TODO */

    /**
     * - Implementiere diskretes Differenzieren als Faltung und
     *   wende es auf ein glattes Testbild an. Was passiert, wenn du ein
     *   verrauschtes Testbild verwendest?
     */
    filter( img0_1, dst, kernelX );
    imshow( "X-Ableitung", dst );
    filter( img0_1, dst, kernelY );
    imshow( "Y-Ableitung", dst );
    waitKey(0);


/* TODO */

    /**
     * - Verwende in der Implementierung nun Faltung mit dem Sobel-Operator
     *   (\code{Sobel}) und beobachte die Ergebnisse auf dem verrauschten
     *   Testbild.
     */
    //sobel-x
    Mat dstX, dstY;
    Sobel(img0_1, dstX, CV_32F, 1, 0 , 3 , 1   );
    imshow( "X-Sobel", dstX );
    Sobel(img0_1, dstY, CV_32F, 0, 1 , 3 , 1   );
    imshow( "Y-Sobel", dstY );
    waitKey(0);
/* TODO */

    /**
     * Aufgabe: Features (10 Punkte)
     *
     * Kanten in Bildern werden häufig als Intensitätssprünge beschrieben.
     *
     * - Berechne den Betrag des Gradienten eines Testbildes und bestimme
     *   Schwellwerte des Gradienten, um möglichst alle Kanten zu entdecken
     *   oder möglichst nur stark ausgeprägte Kanten zu entdecken.
     */
    // die high-Schwellwerte = 0.1*MaxGradient
    // die low-Schwellwete = 0.3*Gradient
    // 1:3 <low-Schwellwete : high-Schwellwerte < 1:2
/* TODO */

    /*
     * - Vergleiche mit dem Ergebnis des Canny-Kantendetektors
     *   (\code{Canny}), wenn er mit diesen Parametern aufgerufen wird.
     */
    img0_1.convertTo(img, CV_8UC1,255);
    Canny(img, dst, 100, 250, 3 );
    imshow( "Canny-1:2.5", dst );
    Canny(img, dst, 100, 300, 3 );
    imshow( "Canny-1:3", dst );
    waitKey(0);
/* TODO */

    /**
     * Einzelne herausragende Punkte werden auch als Featurepunkte oder Ecken
     * bezeichnet, selbst wenn sie nicht auf einer Kante liegen.
     *
     * - Implementiere die Harris-Corner Detektion. Verwende dabei nicht die
     *   OpenCV Methode \code{cornerHarris}, sondern implementiere selbst eine
     *   Funktion, die ein Grauwertbild, einen Parameter $k$ für die Berechnung
     *   des Featureindikators und einen Schwellwert $t$ für ausreichend großen
     *   Ausschlag des Indikators entgegennimmt und die Featurepunkte
     *   zurückgibt.
     */   

     //k = 0.01
    myHarrisCorner(img0_1, dst, 3, 0.01);
    Mat R;
    dst.copyTo(R);
    //t = 0.01
    threshold(dst, dst, 0.01, 255, THRESH_BINARY);
    imshow("myHarrisCorne-R>t:",dst);
/* TODO */

    /**
    * - Zeige die Werte des Detektors vor der Segmentierung mit dem Schwellwert $t$ unter Verwendung
    *   einer geigneten Color Map (\code{applyColorMap}) mit geeigneter Skalierung.
    */
    R.convertTo(R, CV_8UC1, 255);
    applyColorMap(R, R, COLORMAP_OCEAN);
    imshow("myHarrisCorne-R-applyColorMap:",R);
/* TODO */

    /**
     * - Zeichne einen Kreis um jede gefundene Harris-Corner.
     */
    imshow("input-circle-src", src);
    cout<<"src.type:"<<src.type()<<endl;
 //   src.convertTo(src, CV_8UC3);
    cvtColor(src,src, CV_GRAY2BGR);
    for(int j=0; j < dst.rows; j++){
        for(int k = 0; k < dst.cols; k++){
            if( dst.at<float>(j,k) == 255.0 ){
                circle(src, Point(k, j), 5, Scalar(0, 255, 0));


            }
        }

    }
    imshow("myHarrisCorne-R-circle:",src);

/* TODO */
    waitKey(0);
    return 0;
}
