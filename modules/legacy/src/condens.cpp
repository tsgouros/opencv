/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#include <sys/time.h>

#define max(i,j) (i > j)?i:j
#define min(i,j) (i < j)?i:j

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvCreateConDensation
//    Purpose: Creating CvConDensation structure and allocating memory for it
//    Context:
//    Parameters:
//      Kalman     - double pointer to CvConDensation structure
//      DP         - dimension of the dynamical vector
//      MP         - dimension of the measurement vector
//      SamplesNum - number of samples in sample set used in algorithm 
//    Returns:
//    Notes:
//      
//F*/

CV_IMPL CvConDensation* cvCreateConDensation( int DP, int MP, int SamplesNum )
{
    int i;
    CvConDensation *CD = 0;

    if( DP < 0 || MP < 0 || SamplesNum < 0 )
        CV_Error( CV_StsOutOfRange, "" );
    
    /* allocating memory for the structure */
    CD = (CvConDensation *) cvAlloc( sizeof( CvConDensation ));
    /* setting structure params */
    CD->SamplesNum = SamplesNum;
    CD->DP = DP;
    CD->MP = MP;
    /* allocating memory for structure fields */
    CD->flSamples = (float **) cvAlloc( sizeof( float * ) * SamplesNum );
    CD->flNewSamples = (float **) cvAlloc( sizeof( float * ) * SamplesNum );
    CD->flSamples[0] = (float *) cvAlloc( sizeof( float ) * SamplesNum * DP );
    CD->flNewSamples[0] = (float *) cvAlloc( sizeof( float ) * SamplesNum * DP );
    // Set a default value for the scatter range. 
    CD->scatterRange = 0.5;

    /* setting pointers in pointer's arrays */
    for( i = 1; i < SamplesNum; i++ )
    {
        CD->flSamples[i] = CD->flSamples[i - 1] + DP;
        CD->flNewSamples[i] = CD->flNewSamples[i - 1] + DP;
    }

    CD->State = (float *) cvAlloc( sizeof( float ) * DP );

    // This matrix could be reset by a user, but the identity matrix
    // will do the trick for the large majority.
    CD->DynamMatr = (float *) cvAlloc( sizeof( float ) * DP * DP );
    for (int i = 0; i < DP; i++)
      for (int j = 0; j < DP; j++)
	CD->DynamMatr[i * DP + j] = (i == j) ? 1.0 : 0.0 ;

    CD->flConfidence = (float *) cvAlloc( sizeof( float ) * SamplesNum );
    CD->flCumulative = (float *) cvAlloc( sizeof( float ) * SamplesNum );

    CD->RandS = (CvRandState *) cvAlloc( sizeof( CvRandState ) * DP );
    CD->Temp = (float *) cvAlloc( sizeof( float ) * DP );
    CD->RandomSample = (float *) cvAlloc( sizeof( float ) * DP );

    /* Returning created structure */

    return CD;
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvReleaseConDensation
//    Purpose: Releases CvConDensation structure and frees memory allocated for it
//    Context:
//    Parameters:
//      Kalman     - double pointer to CvConDensation structure
//      DP         - dimension of the dynamical vector
//      MP         - dimension of the measurement vector
//      SamplesNum - number of samples in sample set used in algorithm 
//    Returns:
//    Notes:
//      
//F*/
CV_IMPL void
cvReleaseConDensation( CvConDensation ** ConDensation )
{
    CvConDensation *CD = *ConDensation;
    
    if( !ConDensation )
        CV_Error( CV_StsNullPtr, "" );

    if( !CD )
        return;

    /* freeing the memory */
	cvFree( &CD->State );
    cvFree( &CD->DynamMatr);
    cvFree( &CD->flConfidence );
    cvFree( &CD->flCumulative );
    cvFree( &CD->flSamples[0] );
    cvFree( &CD->flNewSamples[0] );
    cvFree( &CD->flSamples );
    cvFree( &CD->flNewSamples );
    cvFree( &CD->Temp );
    cvFree( &CD->RandS );
    cvFree( &CD->RandomSample );
    /* release structure */
    cvFree( ConDensation );
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvConDensUpdateByTime
//    Purpose: Performing Time Update routine for ConDensation algorithm
//    Context:
//    Parameters:
//      Kalman     - pointer to CvConDensation structure
//    Returns:
//    Notes: Modified considerably to fix resampling stage and address
//           usage issues by Tom Sgouros, February 2013.
//      
//F*/
CV_IMPL void
cvConDensUpdateByTime( CvConDensation * ConDens )
{
    int i, j;
    float Sum = 0;
    // These are used in the resampling algorithm.
    struct CvRandState* randState;
    float randNumber;

    // Use this to improve the random number generation.
    struct timeval tv;

    if( !ConDens )
        CV_Error( CV_StsNullPtr, "" );

    /* Sets Temp to Zero */
    icvSetZero_32f( ConDens->Temp, ConDens->DP, 1 );

    /* Calculating the Mean */
    for( i = 0; i < ConDens->SamplesNum; i++ )
    {
        icvScaleVector_32f( ConDens->flSamples[i], ConDens->State, ConDens->DP,
                             ConDens->flConfidence[i] );
        icvAddVector_32f( ConDens->Temp, ConDens->State, ConDens->Temp, ConDens->DP );
        Sum += ConDens->flConfidence[i];
        ConDens->flCumulative[i] = Sum;
    }

    /* Taking the new vector from transformation of mean by dynamics matrix */

    icvScaleVector_32f( ConDens->Temp, ConDens->Temp, ConDens->DP, 1.f / Sum );

    // Note that the DynamMatr *must* be initialized by the calling function.
    // The cvCreateConDensation function does not initialize it.  
    // Initializing it to an identity matrix would be a good choice.
    icvTransformVector_32f( ConDens->DynamMatr, ConDens->Temp, ConDens->State, 
			    ConDens->DP, ConDens->DP );

    // Initialize the random number generator.
    gettimeofday(&tv, NULL);
    cvRandInit( randState, 0, Sum, tv.tv_usec);

    // We want a record of the span of the particle distribution.  The resampled
    // distribution is dependent on this quantity.
    float sampleMax[ConDens->DP], sampleMin[ConDens->DP];
    for (int k = 0; k < ConDens->DP; k++) 
      {
	sampleMax[k] = -1.0e35; sampleMin[k] = 1.0e35;
      }

    /* Updating the set of random samples */
    // The algorithm of the original code always picked the last
    // sample, so was not really a weighted random re-sample.  It
    // wasn't really random, either, due to careless seeding of the
    // random number generation.

    // This version resamples according to the weights calculated by
    // the calling program and tries to be more consistent about
    // seeding the random number generator more carefully.
    for( i = 0; i < ConDens->SamplesNum; i++ )
      {
	// Choose a random number between 0 and the sum of the particles' 
	// weights.
	cvbRand(randState, &randNumber, 1);

	// Use that random number to choose one of the particles.
        j = 0;
        while( (ConDens->flCumulative[j] <= randNumber) && 
	       (j<ConDens->SamplesNum-1))
        {
            j++;
        }

	// Copy the chosen particle.
        icvCopyVector_32f( ConDens->flSamples[j], ConDens->DP, 
			   ConDens->flNewSamples[i] );

	// Keep track of the max and min of the sample particles.
	// We'll use that to calculate the size of the distribution.
	for (int k = 0; k < ConDens->DP; k++) 
	  {
	    sampleMax[k] = max(sampleMax[k], *(ConDens->flNewSamples[i] + k));
	    sampleMin[k] = min(sampleMin[k], *(ConDens->flNewSamples[i] + k));
	  }

      }

    /* Reinitializes the structures to update samples randomly */
    for(int k = 0; k < ConDens->DP; k++ )
      {

	// What's happening here is that the random perturbations used
	// to wiggle the sample points are sized proportionally to the
	// size of the distribution's span.  So if the the samples go
	// from 2 through 5 in some dimension, the random
	// perturbations will go from -1.5 to 1.5.  (So long as the
	// scatterRange parameter is set to 0.5.)
	float diff = ConDens->scatterRange * (sampleMax[k] - sampleMin[k]);

	// This line may not be strictly necessary, but it prevents
	// the particles from congealing into a single particle in the
	// event of a poor choice of fitness (weighting) function.
	diff = max(diff, 0.02 * ConDens->flNewSamples[0][k]);

	// Re-seed the random number generation, and set the limits
	// relative to the geometric extent of the distribution.
	gettimeofday(&tv, NULL);
        cvRandInit( &(ConDens->RandS[k]),
                    -diff, diff,
                    tv.tv_usec + k);
	// We ask for a random number just to give the electronic
	// roulette wheel a good spin.  We're not doing anything with
	// this number.
	cvbRand( ConDens->RandS + k, ConDens->RandomSample + k, 1);
    }


    /* Adding the random-generated vector to every vector in sample set */
    for( i = 0; i < ConDens->SamplesNum; i++ )
    {
        for( j = 0; j < ConDens->DP; j++ )
        {
            cvbRand( ConDens->RandS + j, ConDens->RandomSample + j, 1 );
        }

        icvTransformVector_32f( ConDens->DynamMatr, ConDens->flNewSamples[i],
                                 ConDens->flSamples[i], ConDens->DP, ConDens->DP );
        icvAddVector_32f( ConDens->flSamples[i], ConDens->RandomSample, ConDens->flSamples[i],
	                 ConDens->DP );
    }
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvConDensInitSamplSet
//    Purpose: Performing Time Update routine for ConDensation algorithm
//    Context:
//    Parameters:
//    conDens     - pointer to CvConDensation structure
//    lowerBound  - vector of lower bounds used to random update of sample set
//    lowerBound  - vector of upper bounds used to random update of sample set
//    Returns:
//    Notes:
//      
//F*/

CV_IMPL void
cvConDensInitSampleSet( CvConDensation * conDens, CvMat * lowerBound, CvMat * upperBound )
{
    int i, j;
    float *LBound;
    float *UBound;
    float Prob = 1.f / conDens->SamplesNum;

    if( !conDens || !lowerBound || !upperBound )
        CV_Error( CV_StsNullPtr, "" );

    if( CV_MAT_TYPE(lowerBound->type) != CV_32FC1 ||
        !CV_ARE_TYPES_EQ(lowerBound,upperBound) )
        CV_Error( CV_StsBadArg, "source  has not appropriate format" );

    if( (lowerBound->cols != 1) || (upperBound->cols != 1) )
        CV_Error( CV_StsBadArg, "source  has not appropriate size" );

    if( (lowerBound->rows != conDens->DP) || (upperBound->rows != conDens->DP) )
        CV_Error( CV_StsBadArg, "source  has not appropriate size" );

    LBound = lowerBound->data.fl;
    UBound = upperBound->data.fl;
    /* Initializing the structures to create initial Sample set */
    for( i = 0; i < conDens->DP; i++ )
    {
        cvRandInit( &(conDens->RandS[i]),
                    LBound[i],
                    UBound[i],
                    i );
    }
    /* Generating the samples */
    for( j = 0; j < conDens->SamplesNum; j++ )
    {
        for( i = 0; i < conDens->DP; i++ )
        {
            cvbRand( conDens->RandS + i, conDens->flSamples[j] + i, 1 );
        }
        conDens->flConfidence[j] = Prob;
    }
    /* Reinitializes the structures to update samples randomly */
    for( i = 0; i < conDens->DP; i++ )
    {
        cvRandInit( &(conDens->RandS[i]),
                    (LBound[i] - UBound[i]) / 5,
                    (UBound[i] - LBound[i]) / 5,
                    i);
    }
}
