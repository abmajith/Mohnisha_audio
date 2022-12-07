#include "ivectorComputation.h"

//verified
void initOccuProb(OccuProb *occuprob){
    double **p = (double**) malloc(sizeof(double*) * occuprob->NbSamples);
    if (!p){
        fprintf(stderr, "Failed to allocate %d number of double* memory blocks.\n", occuprob->NbSamples);
        exit(1);
    } else {
        occuprob->resp = p;
    }
    double *q;
    for (int i = 0; i < occuprob->NbSamples; i++){
        q = (double*) malloc(sizeof(double) * occuprob->NC);
        if (!q){
            fprintf(stderr, "Failed to allocate %d numer of double memory blocks.\n", occuprob->NC);
            exit(1);
        } else {
            occuprob->resp[i] = q;
        }
    }
}
//verified
void clearOccuProb(OccuProb *occuprob){
    for (int i = 0; i < occuprob->NbSamples; i++)
        free(occuprob->resp[i]);
    free(occuprob->resp);
}

//verified
void logsumexp(occuprob *prob, double *LSE){
    // LSE should be the size of number of samples in the given prob i.e NbSamples
    double intRes = 0.0f;
    for (int i = 0; i < prob->NbSamples; i++){
        intRes = 0.0f;
        for (int j = 0; j < prob->NC; j++){
            intRes += (double) exp((double) prob->resp[i][j]);
        }
        LSE[i] = log(intRes);
    }
}

//verified
void computeLogWeightProb(MFCCFeatures *mfcc, OccuProb *occuLWprob, double **glbLogDet, double **glbPrec, double **glbMeanPrecProd, double **glbMeanPrecProdSum, double *glbLogWeight){
    double intRes;
    double normlogPi = ((double) -0.5 * nFeatures * log((2.0f * 3.14159265358979323846f)));
    for (int ns = 0; ns < occuLWprob->NbSamples; ns++){
        for (int nc = 0; nc < occuLWprob->NC; nc++){
            intRes = glbMeanPrecProdSum[nc];
            for (int nf = 0; nf < nFeatures; nf++){
                intRes -= ((double) 2.0f * mfcc->Features[ns][nf] * glbMeanPrecProd[nf][nc]);
                intRes += ((double) mfcc->Features[ns][nf] * mfcc->Features[ns][nf] * glbPrec[nc][nf]);
            }
            occuLWprob->resp[ns][nc] = normLogPi - 0.5 * intRes + glbLogDet[nc] + glbLogWeight[nc];
        }
    }
}

//verified
void computeNormWeightProb(OccuProb *occuLWprob){
    //this will calculate the normalized weighted probability of each sample with respect to various components
    int nSamples = occuLWprob->NbSamples;
    double *LSE = (double*) malloc(sizeof(double) * nSamples);
    if (!LSE){
        printf("Could not allocate %d number of double type memory blocks.\n", nSamples);
        exit(1);
    }
    logsumexp(*occuLWprob, LSE);
    double norm, x;
    for (int ns = 0; ns < nSamples; ns++){
        norm = LSE[ns];
        for (int nc = 0; nc < occuLWprob->NC; nc++){
            x = occuLWprob->resp[ns][nc] - norm;
            occuLWprob->resp[ns][nc] = exp(x);
        }
    }
    free(LSE);
}

//verified
void computeZeroStat(OccuProb *occuProb, double *zeroStat){
    //zeroStat is the size of number of components
    double constant = 2.220446049250313e-15f;
    for (int nc = 0; nc < occuProb->NC; nc++){
        zeroStat[nc] = constant;
        for (int ns = 0; ns < occuProb->NbSamples; ns++){
            zeroStat[nc] += occuProb->resp[ns][nc];
        }
    }
}

//verified
void computeFirstStat(OccuProb, *occuProb, double *zeroStat, double **glbSqrtPrec, double **glbMean, double **firstStat, MFCCFeatures *mfcc){
    double intRes;
    for (int nc = 0; nc < occuProb->NC; nc++){
        for (int nf = 0; nf < nFeatures, nf++){
            intRes = 0.0f;
            for (int ns = 0; ns < occuProb->NbSamples; ns++){
                intRes += (occuProb->resp[ns][nc] * mfcc->Features[ns][nf]);
            }
            intRes -= (glbMean[nc][nf] * zeroStat[nc]);
            firstStat[nc][nf] = intRes * glbSqrtPrec[nc][nf];
        }
    }
}

//verified
void computePosteriorPrecMat(double *zeroStat, double **precMatrixL, TVMatrix *tvMat){
    //double intRes;
    int NC = tvMat->NC;
    int nCol = tvMat->TVCol;
    int NF = tvMat->nFeatures;
    for (int r = 0; r < nCol; r++){
        for (int c = 0; c < nCol; c++){
            if (c == r){
                precMatrixL[r][c] = (double) 1.0f;
            }else {
                precMatrixL[r][c] = (double) 0.0f;
            }
            //intRes = 0.0f;
            for (int nc = 0; nc < NC; nc++){
                for (int nf = 0; nf < NF; nf++){
                    precMatrix[r][c] += ((double) zeroStat[nc] * tvMat->TV[(nc * NF) + nf][r] * tvMat->TV[(nc * NF) + nf][c]);
                }
            }
            //precMatrixL[r][c] += intRes;
        }
    }
}

//verified
void computeInversePosteriorPrecMat(double **precMatrixL, double **precInvMatrixL, int len){
    SqMat sqMat;
    sqMat->Row = len;
    initSqMat(&sqMat);
    SqMat invsqMat;
    invsqMat->Row = len;
    initSqMat(&invsqMat);

    for (int i = 0; i < len; i++){
        for (intj = 0; j < len; j++){
            sqMat->matrix[i][j] = precMatrixL[i][j];
        }
    }
    inverseMatrixLU(&sqMat, &invsqMat);
    for (int i = 0; i < len; i++){
        for (int j = 0; j < len; j++){
            precInvMatrixL[i][j] = invsqMat->matrix[i][j];
        }
    }
    clearSqMat(&sqMat);
    clearSqMat(&invsqMat);
}

//verified
void computeWVector(double **precInvMatrixL, TVMatrix *tvMat, double **firstStat, double *wVector){
    int nCol = tvMat->TVCol;
    int NF = tvMat->nFeatures;
    int NC = tvMat->NC;
    int nRow = tvMat->TVrow;
    //allocate intermediate matrix memory
    double **interMatrix = (double**) malloc(sizeof(double*) * nCol);
    if (!interMatrix){
        printf("Could not allocate %d number of double* type memory blocks.\n",nCol);
        exit(1);
    }
    double *p;
    for (int i = 0; i < nRow; i++){
        p = (double*) malloc(sizeof(double) * nRow);
        if (!p){
            printf("Could not allocate %d number of double type memory blocks.\n",nRow);
            exit(1);
        } else {
            interMatrix[i] = p;
        }
    }
    double intRes;
    for (int r = 0; r < nCol; r++){
        for (int c = 0; c < nRow; c++){
            intRes = (double) 0.0f;
            for (int i = 0; i < nCol; i++){
                intRes += precInvMatrixL[r][i] * tvMat->TV[c][i];
            }
            interMatrix[r][c] = intRes;
        }
    }
    double *flattenFirstStat = (double*) malloc(sizeof(double) * nRow);
    if (!flattenFirstStat){
        printf("Could not allocate %d number of double type memory blocks.\n", nRow);
        exit(1);
    }
    for (int nc = 0; nc < NC; nc++){
        for (int f = 0; f < NF; f++){
            flattenFirstStat[NF * nc + f] = firstStat[nc][f];
        }
    }
    for (int k = 0; k < nCol; k++){
        wVector[k] = 0.0f;
        for (int c = 0; c < nRow; c++){
            wVector[k] += interMatrix[k][c] * flattenFirstStat[c];
        }
    }
    //free the intermediate matrix memory blocks
    for (int i = 0; i < nRow; i++){
        free(interMatrix[i]);
    }
    free(interMatrix);
    free(flattenFirstStat);
}

//verified
double computeCosineSimilarityScore(double wTargVector, double wTestVector, int Len){
    double score = (double) 0.0f;
    double wTargNorm = 2.220446049250313e-15f;
    double wTestNorm = 2.220446049250313e-15f;
    for (int i = 0; i < Len; i++){
        score += wTargVector[i] * wTestVector[i];
        wTargNorm += wTargVector[i] * wTargVector[i];
        wTestNorm += wTestVector[i] * wTestVector[i];
    }
    return score / (wTargNorm * wTestNorm);
}
