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

using namespace cv;
using namespace std;
/**
 * Datum: 7.5.2019
 *
 * Übungsblatt: 4
 * Abgabe: 21.5.2019
 */

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <image_left> " << "<image_right>" << std::endl;
        exit(1);
    }

    Mat img1, img2, dst;

    Mat img1_U8 = imread(argv[1]);
    if (img1_U8.empty()) {
        std::cerr << "Could not load image file: " << argv[1] << std::endl;
        exit(1);
    }
    img1_U8.convertTo(img1, CV_32F, 1 / 255.);


    Mat img2_U8 = imread(argv[2]);
    if (img2_U8.empty()) {
        std::cerr << "Could not load image file: " << argv[2] << std::endl;
        exit(1);
    }
    img2_U8.convertTo(img2, CV_32F, 1 / 255.);

    imshow("Input Image left", img1);
    imshow("Input Image right", img2);

    waitKey(0);

    /**
    * Aufgabe: Homographien (5 Punkte)
    *
    * Unter der Annahme, dass Bilder mit einer verzerrungsfreien Lochbildkamera
    * aufgenommen werden, kann man Aufnahmen mit verschiedenen Bildebenen und
    * gleichem Projektionszentren durch projektive Abbildungen, sogenannte
    * Homographien, beschreiben.
    *
    * - Schreibe eine Translation als Homographie auf (auf Papier!).
    * - Verschiebe die Bildebene eines Testbildes um 20 Pixel nach rechts, ohne
    *   das Projektionszentrum zu ändern. Benutze dafür \code{warpPerspective}.
    * - Wieviele Punktkorrespondenzen benötigt man mindestens, um eine projektive
    *   Abbildung zwischen zwei Bildern bis auf eine Skalierung eindeutig zu
    *   bestimmen? Warum? (Schriftlich beantworten!)
    *   4 Punkt
    *
    */

/* TODO */
    Point2f srcQuad[] = {
        Point2f( 0, 0),
        Point2f( img1.cols - 1, 0 ),
        Point2f( img1.cols - 1,  img1.rows - 1 ),
        Point2f( 0, img1.rows - 1 )
    };
    Point2f dstQuad[] = {
        Point2f( 20, 0),
        Point2f( img1.cols - 1, 0 ),
        Point2f( img1.cols - 1,  img1.rows - 1 ),
        Point2f( 20, img1.rows - 1 )
    };
    Mat warp_mat = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective( img1_U8, dst, warp_mat, img1_U8.size()   );
    imshow( " perspective Test ", dst );
    waitKey();


    /**
    * Aufgabe: Panorama (15 Punkte)
    *
    * Ziel dieser Aufgabe ist es, aus zwei gegebenen Bildern ein Panorama zu konstruieren.
    * Dafür muss zunächst aus den gegeben Punktkorrespondenzen:
    *
    * \begin{center}
    * \begin{tabular}{|c|c|}
    * \hline
    * linkes Bild & rechtes Bild \\
    * $(x, y)$ & $(x, y)$ \\ \hline \hline
    * (463, 164) & (225, 179)\\ \hline
    * (530, 357) & (294, 370)\\ \hline
    * (618, 357) &(379, 367)\\ \hline
    * (610, 153) & (369, 168)\\ \hline
    * \end{tabular}
    * \end{center}
    *
    * eine perspektivische Transformation bestimmt werden, mit der die Bilder auf eine
    * gemeinsame Bildebene transformiert werden können.
    *
    * - Berechne die Transformation aus den gegebenen Punktkorrespondenzen.
    *   Benutze die Funktion \code{getPerspectiveTransform}. Welche Dimension
    *   hat der Rückgabewert der Funktion? Warum?
    */

    Point2f srcQuadLeft[] = {
        Point2f( 0, 0),              // top left
        Point2f( 0, img1.rows - 1 ), // bottom left
        Point2f( img1.cols - 1,  img1.rows - 1 ), // bottom right
        Point2f( img1.cols - 1, 0 ) // top right
    };
   Point2f dstQuadLeft[] = {
        Point2f( 463 , 164 ),
        Point2f( 530 , 357 ),
        Point2f( 618 , 357 ),
        Point2f( 610 , 153 )
    };
    warp_mat = getPerspectiveTransform(srcQuadLeft, dstQuadLeft);
    warpPerspective( img1, dst, warp_mat, img1.size()   );
    imshow( " perspective left ", dst );
    waitKey();

    Point2f srcQuadRight[] = {
        Point2f( 0, 0),
        Point2f( 0, img2.rows - 1 ),
        Point2f( img2.cols - 1,  img2.rows - 1 ),
        Point2f( img2.cols - 1, 0  )
    };
   Point2f dstQuadRight[] = {
        Point2f( 225 , 179 ),
        Point2f( 294 , 370 ),
        Point2f( 379 , 367 ),
        Point2f( 369 , 168 )
    };
   Mat dst2;
   Mat warp_mat2 = getPerspectiveTransform(srcQuadRight, dstQuadRight);
   warpPerspective( img2, dst2, warp_mat2, img2.size()   );
   imshow( " perspective right ", dst2 );
   waitKey();

   Mat leftDst;
   Mat H = getPerspectiveTransform( dstQuadLeft, dstQuadRight );
   cout<<"H.type: " << H.type() << endl;
   warpPerspective( img1, leftDst, H, img1.size()   );
   imshow( "new left " , leftDst );
   waitKey();
/* TODO */

    /**
    * - Bestimme die notwendige Bildgröße für das Panoramabild.
    */

    Rect rect;
    rect.x = 463;
    rect.y = 153;
    rect.width = 618 - 463 ;
    rect.height = 357 - 153 ;
    Mat roiLeft = dst(rect);
    Mat roiLeftBorder;
    copyMakeBorder( roiLeft, roiLeftBorder, 0, 370 - 357 , 0, 379 - 225 ,BORDER_CONSTANT, Scalar(0) );
    imshow( "roiLeftBorder", roiLeftBorder );
    waitKey();

    Rect rect2;
    rect2.x = 225;
    rect2.y = 168;
    rect2.width = 379 - 225 ;
    rect2.height = 370 - 168 ;
    Mat roiRight = dst2(rect2);
    Mat roiRightBorder;
    copyMakeBorder( roiRight, roiRightBorder, 0, 168 - 153  , roiLeft.cols - ( 294 - 225 ), 294 - 225 ,BORDER_CONSTANT, Scalar(0) );
    imshow( " roiRightBorder ", roiRightBorder );
    waitKey();
    cout <<"roiLeft: " << roiLeftBorder.size << endl;
    cout <<"roiRihgt: " << roiRightBorder.size << endl;

/* TODO */

    /**
    * - Projiziere das linke Bild in die Bildebene des rechten Bildes. Beachte
    *   dabei, dass auch der linke Bildrand in das Panoramabild projiziert
    *   wird.
    */

    Mat dstout ;
    dstout = roiLeftBorder + roiRightBorder;
    imshow( " roiout ", dstout );
    waitKey();
/* TODO */

    /**
    * - Bilde das Panoramabild so, dass Pixel, für die zwei Werte vorhanden sind,
    *   ihren Mittelwert zugeordnet bekommen.
    */
    int x_start = roiLeft.cols - ( 294 - 225 );
    int x_end = roiLeft.cols;
    cout<< "start: " << x_start << endl;
    cout<< "end: " << x_end << endl;
    cout<< "end.type: " << dstout.type() << endl;

    Mat mask = Mat::zeros(dstout.size(), CV_8UC1);
    vector<vector<Point>> contour;
    vector<Point> pts;
    pts.push_back(Point(x_start + 1, 179 - 168));
    pts.push_back(Point( roiLeft.cols  , 357 - 153));
    pts.push_back(Point(610 - 463 ,0));
    //pts.push_back(Point(50,250));
    contour.push_back(pts);
    drawContours(mask,contour,0,Scalar::all(255),-1);


    Mat ROI, ROI2;
    dstout.copyTo(ROI, mask );
//    imshow( "comman roi ", ROI);
    ROI2 = dstout - ROI;
//    imshow( "comman roi2 ", ROI2);
//    Mat out  = ROI2 + ROI;
//    imshow( "end111", out );

    for( int j = 0; j < ROI.rows; j++  ){
        for( int k = 0; k < ROI.cols ; k++){

                ROI.at<Vec3f>( j, k )[0] *=0.5;
                ROI.at<Vec3f>( j, k )[1] *=0.5;
                ROI.at<Vec3f>( j, k )[2] *=0.5;
        }

    }
//    imshow( "comman roi2 ", ROI);
//    waitKey();


    dstout = ROI2 + ROI;
    imshow( "end", dstout );

    float threshold = 15.0/255;
    for( int j = 0; j < dstout.rows; j++  ){
        for( int k = x_start - 1; k < x_end + 2 ; k++){
            if( dstout.at<Vec3f>( j, k )[0] < threshold
                &&dstout.at<Vec3f>( j, k )[1] <  threshold
                &&dstout.at<Vec3f>( j, k )[2] <  threshold ){

                dstout.at<Vec3f>( j, k )[0] = 0.6+ (dstout.at<Vec3f>( j, k -5 )[0] + dstout.at<Vec3f>( j, k + 5 )[0] )/2;
                dstout.at<Vec3f>( j, k )[1] = 0.6+ (dstout.at<Vec3f>( j, k -5 )[1] + dstout.at<Vec3f>( j, k + 5 )[1] )/2;
                dstout.at<Vec3f>( j, k )[2] = 0.6+ (dstout.at<Vec3f>( j, k -5 )[2] + dstout.at<Vec3f>( j, k + 5 )[2] )/2;
            }

        }

    }

/* TODO */

    /**
    * - Zeige das Panoramabild an.
    */
    imshow( "panorama", dstout );
    waitKey();
/* TODO */

    /**
     * \textit{Hinweis: Die OpenCV High-Level \code{Stitcher}-Klasse ist \textbf{nicht} hilfreich bei der Bearbeitung
     * der Aufgaben. Für ein fertiges Panorama ohne Bearbeitung aller Aufgaben gibt es keine Punkte.}
     */
    return 0;
}
