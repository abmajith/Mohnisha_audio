#ifndef ivectorComputation_H
#define ivectorComputation_H

#include <stdlib.h>
#include <math.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include "frontend_types.h"
#include "gmm_parameters.h"
#include "matrixoperation.h"



typedef struct OccuProb {
    int NbSamples;
    int NC;
    double **resp;
} OccuProb;

typedef struct TVMatrix {
    int NC;
    int TVCol;
    int TVrow;
    int nFeatures;
    double **TV;
} TVMatrix;

//initialize and delete occuprob memory stuff
void initOccuProb(OccuProb *occuprob);
void clearOccuProb(OccuProb *occuprob);
//compute logsumexp of occuprob for normlization
void logsumexp(OccuProb *prob, double *LSE);
//compute log weight prob
void computeLogWeightProb(MFCCFeatures *mfcc, OccuProb *occuLWprob, double *glbLogDet, double **glbPrec, double **glbMeanPrecProd, double *glbMeanPrecProdSum, double *glbLogWeight);
//compute weighted probability i.e occupational probability
void computeNormWeightProb(OccuProb *occuLWprob);
//compute zeroStat of the occupational probability
void computeZeroStat(OccuProb *occuProb, double *zeroStat);
//compute firstStat of the occupational probability
void computeFirstStat(OccuProb *occuProb, double *zeroStat, double **glbSqrtPrec, double **glbMean, double **firstStat, MFCCFeatures *mfcc);
void computeWVector(double **precMatrixInvL, TVMatrix *tvMat, double **firstStat, double *wVector);
void computePosteriorPrecMat(double *zeroStat, double **precMatrixL, TVMatrix *tvMat);
//for calcualting the inverse of a matrix using LU decomposition method
void computeInversePosteriorPrecMat(double **precMatrixL, double **precInvMatrixL, int len);
//for computing cosine simillarity score between target and test ivector stuff
double computeCosineSimilarityScore(double *wTargVector, double *wTestVector, int Len);
#endif
