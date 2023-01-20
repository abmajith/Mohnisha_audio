#include <stdio.h>
#include "frontend_functions.h"
#include "frontend_types.h"
#include <stdlib.h>
#include "gmm_parameters.h"
#include "ivectorComputation.h"
#include<time.h>

int main(){
    clock_t start, end;
    double execution_time;

    RawAudio rawAudio;
    rawAudio.deltaK = 3;
    rawAudio.sampleRate = 44100.0f;
    rawAudio.winLen = 0.02f;
    rawAudio.winStep = 0.01;
    rawAudio.lowFreq = 0.0f;
    rawAudio.highFreq = (double) rawAudio.sampleRate / 2.0f;
    rawAudio.preemph = 0.96f;
    rawAudio.numCept = 19;
    rawAudio.nFilt = 26;
    rawAudio.nFFT = 0;
    rawAudio.cepLifter = 22;
    rawAudio.len = 306176;
    rawAudio.audio = NULL;
    double *p = (double*) malloc(sizeof(double) * 306176);
    if (!p){
        exit(1);
    } else {
        rawAudio.audio= p;
    }

    //for reading wavfile
    FILE *fp;
    fp = fopen("testWav.bin","r");
    if (fp == NULL){
        fprintf(stderr, "testWav binary file can't opened now\n");
        exit(1);
    }
    int count = fread(rawAudio.audio, sizeof(double), rawAudio.len, fp);
    if (count != rawAudio.len){
        printf("Could not read %ld numbers.\n", (long) rawAudio.len);
        exit(1);
    } else {
        printf("Succesfully read the file.\n");
        fclose(fp);
    }
    
    MFCCFeatures Features;
    //for extracting features
    start = clock();
    simpleMFCCExtraction(&rawAudio, &Features);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do feature extraction.\n",execution_time);
    start = clock();
    cmnvNormFeatures(&Features, enMean, enSTD, 38);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do normalized feature extraction.\n",execution_time);
    //printMFCC(&Features);
    free(rawAudio.audio);
    //reading gmmparameteres
    start = clock();
    initEnglishGmmParameters();
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to read gmmParameters.\n",execution_time);
    //initFrenchGmmParameters();
    //reading tvmatrix
    start = clock();
    initEnglishTVmatrix();
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to read tv matrix.\n",execution_time);

    OccuProb occuprob;
    occuprob.NC = 1019;
    occuprob.NbSamples = Features.numberOfFrames;
    initOccuProb(&occuprob);
    //computing weightedprob
    start = clock();
    computeLogWeightProb(&Features, &occuprob, enLogDet, enPrec, enMeanPrecProd, enMeanPrecProdSum, enLogWeight);
    computeNormWeightProb(&occuprob);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute norm weighted probability.\n",execution_time);
    double *zeroStat = (double*) malloc(sizeof(double) * occuprob.NC);
    //computing zeroStat
    start = clock();
    computeZeroStat(&occuprob, zeroStat);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute zero stat.\n",execution_time);


    double **firstStat = (double**) malloc(sizeof(double*) * occuprob.NC);
    for (int i = 0; i < occuprob.NC; i++)
        firstStat[i] = (double*) malloc(sizeof(double) * nFeatures);
    //computing firstStat
    start = clock();
    computeFirstStat(&occuprob, zeroStat, enSqrtPrec, enGMMMean, firstStat, &Features);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute firstStat.\n",execution_time);
    double **precMatrixL = (double**) malloc(sizeof(double*) * enTVCol);
    double **precInvMatrixL = (double**) malloc(sizeof(double*) * enTVCol);
    for (int i = 0; i < enTVCol; i++){
        precMatrixL[i] = (double*) malloc(sizeof(double) * enTVCol);
        precInvMatrixL[i] = (double*) malloc(sizeof(double) * enTVCol);
    }
    TVMatrix tvMat;
    tvMat.NC = enNC;
    tvMat.TVCol = enTVCol;
    tvMat.TVrow = enNC * nFeatures;
    tvMat.nFeatures = nFeatures;
    tvMat.TV = enTVMat;
    ///computing precmat
    start = clock();
    computePosteriorPrecMat(zeroStat, precMatrixL, &tvMat);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute posterior precmat.\n",execution_time);
    /*for (int i = 0; i < 150; i++){
        for (int j = 0; j < 150; j++){
            printf("(%lf,%lf)\t",precMatrixL[i][j],precInvMatrixL[i][j]);
        }
        printf("\n");
    }*/
    //computing inverse precision matrix
    start = clock();
    computeInversePosteriorPrecMat(precMatrixL, precInvMatrixL, enTVCol);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute inverse precision matrix.\n",execution_time);
    double *wVector = (double*) malloc(sizeof(double) * enTVCol);
    start = clock();
    computeWVector(precInvMatrixL, &tvMat,firstStat,wVector);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to compute iVector.\n",execution_time);
    //for (int i = 0; i < enTVCol; i++)
        //fprintf(stderr,"%lf, ", wVector[i]);
    printf("\n\n");
    free(wVector);
    free(zeroStat);
    /*
    for (int i = 0; i < enTVCol; i++){
        for (int j = 0; j < enTVCol; j++){
            printf("(%lf,%lf) ", precMatrixL[i][j], precInvMatrixL[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    
    for (int i = 0; i < nFeatures; i++)
        printf("%lf\t",firstStat[0][i]);
    printf("\n\n");
    for (int i = 0; i < occuprob.NC; i++){
        free(firstStat[i]);
    }
    */
    
    for (int i = 0; i < enTVCol; i++){
        free(precMatrixL[i]);
        free(precInvMatrixL[i]);
    }
    free(precMatrixL);
    free(precInvMatrixL);
    free(firstStat);
    clearOccuProb(&occuprob);
    clearEnglishGmmParameters();
    //clearFrenchGmmParameters();
    clearEnglishTVmatrix();
    delMFCCFeaturesMem(&Features);
    return 1;
}
