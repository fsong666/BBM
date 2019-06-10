/**
 * Bildbasierte Modellierung SS 2019
 * Prof. Dr.-Ing. Marcus Magnor
 *
 * Betreuer: JP Tauscher (tauscher@cg.cs.tu-bs.de)
 * URL: https://graphics.tu-bs.de/teaching/ss19/bbm
 */


#include <iostream>
#include <set>

#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
/**
 * Datum: 21.5.2019
 *
 * Übungsblatt: 5
 * Abgabe: 28.5.2019
 */

int modelPoints = 4;
CvSize modelSize( 3, 3);
int maxBasicSolutions = 1;
bool checkPartialSubsets = false;
CvRNG rng = cvRNG(-1);


void printMat(CvMat *matrix, bool save_or_show =false,FILE *fp=NULL)
{
    int i=0;
    int j=0;
    cout<<"[";
    for(i=0;i < matrix->rows;i++)//行
    {
        if (save_or_show)
        {
            fprintf(fp,"\n");
        }
        else
        {
            printf("\n");
        }
        switch(matrix->type&0X07)
        {
        case CV_32F:
        case CV_64F:
            {
                for(j=0;j<matrix->cols;j++)//列
                {
                    if (save_or_show)
                    {
                        fprintf(fp,"%9.2f ",(float)cvGetReal2D(matrix,i,j));
                    }
                    else
                    {
                        printf("%9.2f ",(float)cvGetReal2D(matrix,i,j));
                    }
                }
                break;
            }
        case CV_8U:
        case CV_16U:
            {
                for(j=0;j<matrix->cols;j++)
                {
                    printf("%6d  ",(int)cvGetReal2D(matrix,i,j));
                    if (save_or_show)
                    {
                        fprintf(fp,"%6d  ",(int)cvGetReal2D(matrix,i,j));
                    }
                    else
                    {
                        printf("%6d  ",(int)cvGetReal2D(matrix,i,j));
                    }
                }
                break;
            }
        default:
            break;
        }
    }
    cout<<"]"<< endl;
}

// Nutze hier deine Funktion aus Übungsbatt 4. Du darfst auch die Musterlösung verwenden.
void createPanorama(const Mat &img1, const Mat &img2, Mat H, Mat &panorama) {

    vector<Point2f> corners = {{0.f,               0.f},
                              {(float) img1.cols,  0.f},
                              {0.f,               (float) img1.rows},
                              {(float) img1.cols, (float) img1.rows}};
    perspectiveTransform(corners, corners, H);

    int max_x = static_cast<int>(std::max(corners[1].x, corners[3].x) + .5f);
    int min_x = static_cast<int>(std::min(corners[0].x, corners[2].x));
    int max_y = static_cast<int>(std::max(corners[2].y, corners[3].y) + .5f);
    int min_y = static_cast<int>(std::min(corners[0].y, corners[1].y));

    int offset_x = -std::min(0, min_x);
    int offset_y = -std::min(0, min_y);

    Size pano_size(std::max(max_x, img2.cols) + offset_x, std::max(max_y, img2.rows) + offset_y);

    H.convertTo(H, CV_32FC1);

    Mat offset = Mat::eye(3, 3, CV_32F);
    offset.at<float>(0, 2) = offset_x;
    offset.at<float>(1, 2) = offset_y;

    Mat trans = offset * H;

    Mat left, right;
    warpPerspective( img1, left,  trans,   pano_size );
    warpPerspective( img2, right, offset,  pano_size );
//    imshow("left", left);
 //   imshow("right", right);

    Mat mask_l, mask_r;
    threshold( left,  mask_l, 0, 1, CV_THRESH_BINARY );
    threshold( right, mask_r, 0, 1, CV_THRESH_BINARY );
    add(left, right, panorama);
    add( mask_l, mask_r, mask_l );
    divide( panorama, mask_l, panorama );
}

/**
 * Aufgabe: Features (10 Punkte)
 *
 * Featuredeskriptoren finden und beschreiben lokale Merkmale in Bildern. Diese Merkmale
 * sind markant und können idealerweise in anderen Bildern der selben Szene wiedergefunden werden.
 *
 * In der Vorlesung betrachten wir den SIFT-Algorithmus. SIFT ist durch ein US-Patenent geschützt und daher nur
 * in der OpenCV Extension \code{nonfree} enthalten. Diese ist im CIP-Pool nicht installiert. OpenCV bietet jedoch eine
 * Reihe weiterer Featuredeskriptoren. Mit ORB stellt OpenCV eine eigens entwickelte freie Alternative zu SIFT zur
 * Verfügung.
 *
 * - Berechne mit einem Featuredeskriptor deiner Wahl Featurepunkte der Testbilder.
 * - Zeichne die Features in die Testbilder ein.
 * - Implementiere das Finden von Korrespondenzen zwischen Paaren von Testbildern (ohne Verwendungung einer
 *   OpenCV \code{DescriptorMatcher} Klasse).
 *   Aus der Vorlesung wissen wir, dass ein Match nur akzeptiert werden soll, wenn die beste Deskriptordistanz
 *   kleiner ist als das 0.6-fache der zweitbesten Deskriptordistanz. Implementiere diese Prüfung. Was passiert,
 *   wenn der Faktor auf 0.4 bzw. 0.8 gesetzt wird?
 * - Visualisiere die Korrespondenzen in geeigneter Form.
 */


bool cmp(Scalar const &a, Scalar const &b) {
    return a.val[0]< b.val[0];// && a.val[1] < b.val[1] && a.val[2] < b.val[2] ;
}
// Matches a descriptor desc against a list of descriptors desclist with an acceptance ratio ratio.
// Returns the id of the match in featlist or -1
int matchDescriptors(const Mat &desc, const Mat &desclist, float ratio) {
    int rows = desclist.rows;
    vector<float> distanceList;
   // vector<Scalar> distanceList;
    Mat distanceMat;
    Scalar sum;
    float sumAllChannels = 0;

    //distanceList for descList herstellen
    for( int j = 0; j < rows; j++  ){
        Mat objectDesc = desclist.row(j);
        multiply(desc - objectDesc, desc - objectDesc, distanceMat );
        sum = cv::sum( distanceMat );
        sumAllChannels =(float)( sum[0] + sum[1] + sum[2] );//scalr[num, 0, 0 ,0]
        distanceList.push_back(sqrt( sumAllChannels) );
        cout<<"scalr " << sum << endl;
    }
    auto min = std::min_element(std::begin(distanceList), std::end(distanceList) );
    int minIndex = distance( begin(distanceList), min );

    std::sort(distanceList.begin(), distanceList.end() );

    //die beste Deskriptordistanz kleiner ist als das 0.6-fache der zweitbesten Deskriptordistanz
    if( distanceList[0]< ratio * distanceList[1]  ){
        return minIndex;
    }else{
        return -1;
    }
}

// Finds matches between two lists of descriptors desclist1 and desclist2 with an acceptance ratio ratio.
// Returns a pairs of ids ptpairs.
void findMatches(const Mat &desclist1, Mat &desclist2,
                 vector<std::pair<unsigned int, unsigned int>> &ptpairs, float ratio) {

    for(int j = 0; j < desclist1.rows; j++){

        Mat desc = desclist1.row(j);
        int matchId = matchDescriptors( desc, desclist2 , ratio);
        if( matchId >= 0 ){
            pair<unsigned int, unsigned int> pair( (unsigned int)j, (unsigned int)matchId );
            ptpairs.push_back( pair );
        }
    }
}

/**
 * Aufgabe: RANSAC (10 Punkte)
 *
 * Tatsächlich liefern selbst sehr gute Matching-Algorithmen stets einige
 * Korrespondenzen, die nicht denselben 3D Punkt beschreiben. Homographien, die
 * auf diesen Korrespondenzen beruhen, liefern nicht die gewünschten
 * Transformationen. Der RANSAC-Algorithmus versucht, aus der Gesamtmenge der
 * Korrespondenzen diejenigen herauszusuchen, die zueinander konsistent sind.
 *
 * - Implementiere selbständig den RANSAC-Algorithmus. Verwende als Qualitätsmaß den
 *   Abstand zwischen den Featurepunkten des einen Bildes und den
 *   transformierten Featurepunkten des anderen Bildes: Abweichungen von mehr
 *   als vier Pixeln in einer Koordinate kennzeichnen einen Punkt als
 *   Outlier. Bei gleicher Anzahl konsistenter Korrespondenzen entscheidet
 *   die Gesamtabweichung aller gültigen Korrespondenzen.
 */
bool checkSubset( const CvMat* m, int count )
{
    int j, k, i, i0, i1;
    CvPoint2D64f* ptr = (CvPoint2D64f*)m->data.ptr;

    assert( CV_MAT_TYPE(m->type) == CV_64FC2 );

    if( checkPartialSubsets )
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for( i = i0; i <= i1; i++ )
    {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for( j = 0; j < i; j++ )
        {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for( k = 0; k < j; k++ )
            {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if( k < j )
                break;
        }
        if( j < i )
            break;
    }

    return i >= i1;
}

bool getSubset( const CvMat* m1, const CvMat* m2,
                                   CvMat* ms1, CvMat* ms2, int maxAttempts )  //maxAttempts set 300
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
    int type = CV_MAT_TYPE(m1->type), elemSize = CV_ELEM_SIZE(type);
    const int *m1ptr = m1->data.i, *m2ptr = m2->data.i;
    int *ms1ptr = ms1->data.i, *ms2ptr = ms2->data.i;
    int count = m1->cols*m1->rows;

    assert( CV_IS_MAT_CONT(m1->type & m2->type) && (elemSize % sizeof(int) == 0) );
    elemSize /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            //Generate random numbers within count, count is the length of the sequence
            idx[i] = idx_i = cvRandInt(&rng) % count;

            //Ensure that the generated random numbers are not duplicated
            for( j = 0; j < i; j++ )
                if( idx_i == idx[j] )
                    break;
            if( j < i )
                continue;
            for( k = 0; k < elemSize; k++ )
            {
                //Give randomly generated numbers to ms1 and ms2
                ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
                ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
            }
            if( checkPartialSubsets && (!checkSubset( ms1, i+1 ) || !checkSubset( ms2, i+1 )))
            {
                iters++;
                continue;
            }
            i++;
        }
        if( !checkPartialSubsets && i == modelPoints &&
            (!checkSubset( ms1, i ) || !checkSubset( ms2, i )))
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}

/**
 * @brief runKernel with 4 match pairs obtain the homography
 * @param m1 input pointer for matched Matrix of src keypoints
 * @param m2 input pointer for matched Matrix of obj keypoints
 * @param H output pointer for homography matrix
 * @return 1 as success run, 0 as failed
 */
int runKernel( const CvMat* m1, const CvMat* m2, CvMat* H )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

    double LtL[9][9], W[9][1], V[9][9];
    CvMat _LtL = cvMat( 9, 9, CV_64F, LtL );
    CvMat matW = cvMat( 9, 1, CV_64F, W );
    CvMat matV = cvMat( 9, 9, CV_64F, V );
    CvMat _H0 = cvMat( 3, 3, CV_64F, V[8] );
    CvMat _Htemp = cvMat( 3, 3, CV_64F, V[7] );
    CvPoint2D64f cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count; cm.y /= count;
    cM.x /= count; cM.y /= count;

    for( i = 0; i < count; i++ )
    {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y);
    }

    if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
        fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
        return 0;
    sm.x = count/sm.x; sm.y = count/sm.y;
    sM.x = count/sM.x; sM.y = count/sM.y;

    double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
    double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
    CvMat _invHnorm = cvMat( 3, 3, CV_64FC1, invHnorm );
    CvMat _Hnorm2 = cvMat( 3, 3, CV_64FC1, Hnorm2 );

    cvZero( &_LtL );
    for( i = 0; i < count; i++ )
    {
        double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
        double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
        double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
        double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
        int j, k;
        for( j = 0; j < 9; j++ )
            for( k = j; k < 9; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
    }
    cvCompleteSymm( &_LtL );

    //cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    cvEigenVV( &_LtL, &matV, &matW );
    cvMatMul( &_invHnorm, &_H0, &_Htemp );
    cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
    cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

    return 1;
}

bool refine( const CvMat* m1, const CvMat* m2, CvMat* model, int maxIters )
{
    CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
    int i, j, k, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    CvMat modelPart = cvMat( solver.param->rows, solver.param->cols, model->type, model->data.ptr );
    cvCopy( &modelPart, solver.param );

    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;

        if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
            break;

        for( i = 0; i < count; i++ )
        {
            const double* h = _param->data.db;
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            double err[] = { _xi - m[i].x, _yi - m[i].y };
            if( _JtJ || _JtErr )
            {
                double J[][8] =
                {
                    { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                    { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                };

                for( j = 0; j < 8; j++ )
                {
                    for( k = j; k < 8; k++ )
                        _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                    _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                }
            }
            if( _errNorm )
                *_errNorm += err[0]*err[0] + err[1]*err[1];
        }
    }

    cvCopy( solver.param, &modelPart );
    return true;
}
/**
 * @brief computeReprojError
 * @param m1
 * @param m2
 * @param model input pointer for homography matrix as model
 * @param _err
 */
void computeReprojError( const CvMat* m1, const CvMat* m2,
                                                const CvMat* model, CvMat* _err )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    const double* H = model->data.db;
    float* err = _err->data.fl;

    for( i = 0; i < count; i++ )
    {
        double ww = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double dx = (H[0]*M[i].x + H[1]*M[i].y + H[2])*ww - m[i].x;
        double dy = (H[3]*M[i].x + H[4]*M[i].y + H[5])*ww - m[i].y;
        err[i] = (float)(dx*dx + dy*dy);
    }
}

int findInliers( const CvMat* m1, const CvMat* m2,
                                    const CvMat* model, CvMat* _err,
                                    CvMat* _mask, double threshold )
{
    int i, count = _err->rows*_err->cols, goodCount = 0;
    const float* err = _err->data.fl;
    uchar* mask = _mask->data.ptr;

    //3.  Calculate the projection error of all data in the dataset and the model H.
    computeReprojError( m1, m2, model, _err );  //_err is matrix of projection error

    threshold *= threshold;
    for( i = 0; i < count; i++ )
    {   //4.  If the error is less than the threshold, add the inner point set.
        mask[i] = ( err[i] <= threshold );
        goodCount += (uchar)mask[i];// = err[i] <= threshold;//goodCount is numer of Inliers and get from err[i]
    }
    cout<<"mask[10]: " << mask[10]<< endl;
    return goodCount;
}

bool runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
                                    CvMat* mask0, double reprojThreshold,
                                    double confidence, int maxIters )
{
    bool result = false;
    cv::Ptr<CvMat> mask = cvCloneMat(mask0);   //mark Matrix, mark Inliers as 1 and Outliers as 0
    cv::Ptr<CvMat> models, err, tmask;
    cv::Ptr<CvMat> ms1, ms2;

    int iter, niters = maxIters;
    int count = m1->rows*m1->cols, maxGoodCount = 0; // count is total numer of match pairs
    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

    if( count < modelPoints )  //RANSAC，modelPoints is 4
        return false;

    // create a Mat for H  3*3 ; CvSize modelSize( 3, 3)
    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err = cvCreateMat( 1, count, CV_32FC1 );
    tmask = cvCreateMat( 1, count, CV_8UC1 );
    cout<<"cout: "<< count<< endl;
    if( count > modelPoints )
    {
        ms1 = cvCreateMat( 1, modelPoints, m1->type ); // a row vector with 4 points from match List m1
        ms2 = cvCreateMat( 1, modelPoints, m2->type ); // a row vector with 4 points from match List m2
    }
    else
    {
        niters = 1;
        ms1 = cvCloneMat(m1);
        ms2 = cvCloneMat(m2);
    }

    for( iter = 0; iter < niters; iter++ )
    {
        int  goodCount, nmodels;

        if( count > modelPoints )
        {
            //1.  to randomly select two vector with 4 points from the match List m1 and m2,
            //in order to calculate the homography matrix H for the next step
            bool found = getSubset( m1, m2, ms1, ms2, 300 );//300 is iters
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        //2.  calculate out the homography matrix as models 3*3
        nmodels = runKernel( ms1, ms2, models );

        cout<<"--------"<<endl;
        cout<<"model H:\n";
        printMat( models );
        cout<<"--------"<<endl;

        if( nmodels <= 0 )
        {
             continue;
        }else{
            //3.   Calculate the projection error  and  4. find out Inliers
            goodCount = findInliers( m1, m2, models, err, tmask, reprojThreshold );

            cout<<"--------\n err: "<<endl;
            printMat( err);
            cout<<"--------\n tmask: "<<endl;
            printMat( tmask);
            cout<<"iter: "<< iter << " niters: "
               << niters << " goodCount: "
               << goodCount<<" maxGoodCount: "
               << maxGoodCount<<endl;

            //5. If the current number of inner point set is greater than the best optimal inner point
            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
                std::swap(tmask, mask);
                cvCopy( models, model );

                //update the best number of inner points
                maxGoodCount = goodCount;

                //update niters
                niters = cvRANSACUpdateNumIters( confidence,
                    (double)(count - goodCount)/count, modelPoints, niters );
            }
        }

    }

    if( maxGoodCount > 0 )
    {
        if( mask != mask0 )
            cvCopy( mask, mask0 );
        result = true;
    }

    return result;

}

template<typename T> int icvCompressPoints(T* ptr, const uchar* mask, int mstep, int count)
{
    int i, j;
    for (i = j = 0; i < count; i++)
    if (mask[i*mstep])
    {
        if (i > j)
            ptr[j] = ptr[i];
        j++;
    }
    return j;
}
void findHomographyRANSAC(const unsigned int &ransac_iterations,
                        const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
                         const vector<std::pair<unsigned int, unsigned int>> &matches, Mat &H) {
    vector<Point2f> match_kpts1;
    vector<Point2f> match_kpts2;
    for(auto p : matches ){
        Point2f p_left = keypoints1[p.first].pt ;
        Point2f p_right = keypoints2[p.second].pt ;
        match_kpts1.push_back( p_left );
        match_kpts2.push_back( p_right );
    }
    double ransacReprojThreshold = 4;
    //Mat mask;
//    H = findHomography( Mat(match_kpts1), Mat(match_kpts2), CV_RANSAC, ransacReprojThreshold , mask, ransac_iterations);

    const double confidence = 0.995;
    const int maxIters = ransac_iterations;
    CvMat _pt1 = Mat(match_kpts1);
    CvMat _pt2 = Mat(match_kpts2);
    CvMat* objectPoints = &_pt1;
    CvMat* imagePoints  = &_pt2;

    H.create(3, 3, CV_64F);
    CvMat _H = H;
    CvMat* __H = &_H;

    double H1[9];
    CvMat matH = cvMat( 3, 3, CV_64FC1, H1 );
    int count = MAX(imagePoints->cols, imagePoints->rows);;

    OutputArray _mask = noArray();
    CvMat* mask= 0;
    CvMat c_mask;
    Ptr<CvMat> m, M, tempMask;
    bool result = true;


    CV_Assert( CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints) );

    count = matches.size();
    CV_Assert( count >= 4 );

    M = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( &_pt1, M );

    m = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( &_pt2, m );

    if( _mask.needed() )
    {
            _mask.create(count, 1, CV_8U, -1, true);
            mask = &(c_mask = _mask.getMat());
    }

    if( mask )
    {
        CV_Assert( CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
        (mask->rows == 1 || mask->cols == 1) &&
        mask->rows*mask->cols == count );
    }
    if( mask || count > 4 )
         tempMask = cvCreateMat( 1, count, CV_8U );
    if( !tempMask.empty() )
           cvSet( tempMask, cvScalarAll(1.) );

    result = runRANSAC( M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);

    /**
     * - Berechne mit allen von dir bestimmmten gültigen Korrespondenzen eine Homographie zwischen den Bildern.
     * - Stelle mit dieser Homographie ein Panorama her.
     */

    if( result && count > 4 )
    {
        //Compression, making the sequence compact
        icvCompressPoints( (CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count );
        count = icvCompressPoints( (CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count );
        M->cols = m->cols = count;    //After screening, this count is the number of inner points.

        runKernel( M, m, &matH );
        refine( M, m, &matH, 10 );
    }

    cvConvert( &matH, __H);

}

int main(int argc, char **argv) {
//int main(void) {
    if (argc <= 3) {
        std::cerr << "Usage: main <image-file-name1> <image-file-name2> <ratio>" << std::endl;
        exit(0);
    }

    Mat img1, img2;

    Mat img1_U8 = imread(argv[1], IMREAD_COLOR);
    Mat img2_U8 = imread(argv[2], IMREAD_COLOR);

    if (img1_U8.empty() || img2_U8.empty()) {
        std::cout << "Could not load image file: " << argv[1] << " or " << argv[2] << std::endl;
        exit(0);
    }

    float ratio = atof(argv[3]);

    img1_U8.convertTo(img1, CV_32F, 1 / 255.);
    img2_U8.convertTo(img2, CV_32F, 1 / 255.);

    imshow("Input Image left", img1_U8);
    imshow("Input Image right", img2_U8);
    waitKey(0);

    //parameters for orb detector
    int   nfeatures = atoi(argv[4]);//400
    float scaleFactor =1.2f;
    int  	nlevels = 8;


    // create keypoint and descriptor list
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<ORB> orb = ORB::create( nfeatures, scaleFactor, nlevels);

    //detect orb features
    orb->detect(img1_U8, kpts1);
    orb->detect(img2_U8, kpts2);

    //compute orb descriptors
    orb->compute(img1_U8, kpts1, desc1);// a vector with 32 dimensons , je elemtne ist die Frequenz of Orietierung
    orb->compute(img2_U8, kpts2, desc2);//
    cout<<"desc1: " << desc1.size<< endl;
    cout<<"desc2: " << desc2.size<< endl;

//    Mat leftKeypoints;
//    //img muss CV_8U OR CV_8UC3
//    drawKeypoints(img1_U8, kpts1, leftKeypoints);
//    imshow("left_keypoints", leftKeypoints );
//    Mat rightKeypoints;
//    drawKeypoints(img2_U8, kpts2, rightKeypoints);
//    imshow("right_keypoints", rightKeypoints );


    //match keypoints
    vector<std::pair<unsigned int, unsigned int>> pairs;
    vector<DMatch> newMatches;
    vector<Point2f> match_kpts1;
    vector<Point2f> match_kpts2;

    findMatches(desc1, desc2, pairs, ratio );

    if( !pairs.empty()){
        for(auto tempMatch : pairs ){

            DMatch dm(tempMatch.first, tempMatch.second,  0);
            newMatches.push_back( dm );
            Point2f p_left = kpts1[tempMatch.first].pt ;
            Point2f p_right = kpts2[tempMatch.second].pt ;
            match_kpts1.push_back( p_left );
            match_kpts2.push_back( p_right );
            cout<<"my-srcId: " << tempMatch.first <<"  point: "<<p_left << endl;
            cout<<"my_objId: " << tempMatch.second<<"  point: "<<p_right<<"\n"<< endl;
        }
    }

    Mat matcher_img;
    drawMatches(img1_U8, kpts1, img2_U8, kpts2, newMatches, matcher_img);
    imshow("my Matcher", matcher_img);

    int iters = 2000;
    Mat Hnew;
    findHomographyRANSAC(iters, kpts1, kpts2, pairs, Hnew);

    Mat  panorama2;
    createPanorama( img1, img2, Hnew, panorama2);
    imshow("panorama", panorama2);
    waitKey(0);

    cout<<"my match.size: " << newMatches.size() << endl;
    return 0;
    /**
     * \textit{Hinweis: Die OpenCV High-Level \code{Stitcher}-Klasse ist \textbf{nicht} hilfreich bei der Bearbeitung
     * der Aufgaben. Für ein fertiges Panorama ohne Bearbeitung aller Aufgaben gibt es keine Punkte.}
     */

}

//    Mat Hx = findHomography( Mat(match_kpts1), Mat(match_kpts2), CV_RANSAC,  ransacReprojThreshold );
//    vector<Point2f> points_left = {{463, 164},
//                                    {530, 357},
//                                    {618, 357},
//                                    {610, 153}};
//    vector<Point2f> points_right = {{225, 179},
//                                     {294, 370},
//                                     {379, 367},
//                                     {369, 168}};
//    Mat H = getPerspectiveTransform(points_left, points_right);
//    Mat panorama1;
//    createPanorama( img1, img2, H, panorama1);
//    imshow("panorama without RANSAC", panorama1);
