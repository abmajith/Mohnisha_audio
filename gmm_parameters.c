#include "gmm_parameters.h"
#include <stdlib.h>

const int frNC = 1023;
const int enNC = 1019;


const char frweightFile[]          = "frWeights.bin";
const char frmeanFile[]            = "frMean.bin";
const char frcovarFile[]           = "frCovar.bin";
const char frlogdetFile[]          = "frLogDet.bin";
const char frlogweightFile[]       = "frLogWeights.bin";
const char frmeanprecprodFile[]    = "frMeanPrecProd.bin";
const char frmeanprecprodsumFile[] = "frMeanPrecProdSum.bin";
const char frprecFile[]            = "frPrec.bin";
const char frsqrtprecFile[]        = "frSqrtPrec.bin";



const char enweightFile[]          = "enWeights.bin";
const char enmeanFile[]            = "enMean.bin";
const char encovarFile[]           = "enCovar.bin";
const char enlogdetFile[]          = "enLogDet.bin";
const char enlogweightFile[]       = "enLogWeights.bin";
const char enmeanprecprodFile[]    = "enMeanPrecProd.bin";
const char enmeanprecprodsumFile[] = "enMeanPrecProdSum.bin";
const char enprecFile[]            = "enPrec.bin";
const char ensqrtprecFile[]        = "enSqrtPrec.bin";



double *frGMMWeights      = NULL;//(double*) malloc(sizeof(double) * frNC);
double **frGMMMean        = NULL;//(double**) malloc(sizeof(double*) * frNC);
double **frGMMCovar       = NULL;//(double**) malloc(sizeof(double*) * frNC);
double *frLogDet          = NULL;//(double*) malloc(sizeof(double) * frNC);
double **frPrec           = NULL;//(double**) malloc(sizeof(double*) * frNC);
double **frSqrtPrec       = NULL;//(double**) malloc(sizeof(double*) * frNC);
double *frLogWeight       = NULL;//(double*) malloc(sizeof(double) * frNC);
double *frMeanPrecProdSum = NULL;//(double*) malloc(sizeof(double) * frNC);
double **frMeanPrecProd   = NULL;//(double**) malloc(sizeof(double*) * nFeatures);


double *enGMMWeights      = NULL;//(double*) malloc(sizeof(double) * enNC);
double **enGMMMean        = NULL;//(double**) malloc(sizeof(double*) * enNC);
double **enGMMCovar       = NULL;//(double**) malloc(sizeof(double*) * enNC);
double *enLogDet          = NULL;//(double*) malloc(sizeof(double) * enNC);
double **enPrec           = NULL;//(double**) malloc(sizeof(double*) * enNC);
double **enSqrtPrec       = NULL;//(double**) malloc(sizeof(double*) * enNC);
double *enLogWeight       = NULL;//(double*) malloc(sizeof(double) * enNC);
double *enMeanPrecProdSum = NULL;//(double*) malloc(sizeof(double) * enNC);
double **enMeanPrecProd   = NULL;//(double**) malloc(sizeof(double*) * nFeatures);



const int frTVCol = 150;
const int enTVCol = 150;

const char frTVMatFile[] = "frenchTVmat.bin";
const char enTVMatFile[] = "englishTVmat.bin";

double **frTVMat = NULL;
double **enTVMat = NULL;



void initEnglishGmmParameters(){
    FILE *fp;
    //for opening the weights
    fp = fopen(enweightFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", enweightFile);
        exit(1);
    }
    enGMMWeights = (double*) malloc(sizeof(double) * enNC);
    if (!enGMMWeights){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", enNC);
        exit(1);
    }
    int count = fread(enGMMWeights, sizeof(double), enNC, fp);
    if (count != enNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", enNC, enweightFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", enweightFile);
    }
    fclose(fp);

    //for opening logweights
    fp = fopen(enlogweightFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", enlogweightFile);
        exit(1);
    }
    enLogWeight = (double*) malloc(sizeof(double) * enNC);
    if (!enLogWeight){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", enNC);
        exit(1);
    }
    count = fread(enLogWeight, sizeof(double), enNC, fp);
    if (count != enNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", enNC, enlogweightFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", enlogweightFile);
    }
    fclose(fp);

    //for opening the MeanPrecProdSum
    fp = fopen(enmeanprecprodsumFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", enmeanprecprodsumFile);
        exit(1);
    }
    enMeanPrecProdSum = (double*) malloc(sizeof(double) * enNC);
    if (!enMeanPrecProdSum){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", enNC);
        exit(1);
    }
    count = fread(enMeanPrecProdSum, sizeof(double), enNC, fp);
    if (count != enNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", enNC, enmeanprecprodsumFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", enmeanprecprodsumFile);
    }
    fclose(fp);

    //for opening the logDeterminant file enlogdetFile
    fp = fopen(enlogdetFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", enlogdetFile);
        exit(1);
    }
    enLogDet = (double*) malloc(sizeof(double) * enNC);
    if (!enLogDet){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", enNC);
        exit(1);
    }
    count = fread(enLogDet, sizeof(double), enNC, fp);
    if (count != enNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", enNC, enlogdetFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", enlogdetFile);
    }
    fclose(fp);


    //for opening the mean file
    fp = fopen(enmeanFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", enmeanFile);
        exit(1);
    }
    enGMMMean  = (double**) malloc(sizeof(double*) * enNC);
    if (!enGMMMean){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", enNC);
        exit(1);
    }
    for (int i = 0; i < enNC; i++){
        enGMMMean[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!enGMMMean[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(enGMMMean[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, enmeanFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succesfully read the %s file.\n",enmeanFile);
    fclose(fp);

    //for opening the covar file
    fp = fopen(encovarFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", encovarFile);
        exit(1);
    }
    enGMMCovar  = (double**) malloc(sizeof(double*) * enNC);
    if (!enGMMCovar){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", enNC);
        exit(1);
    }
    for (int i = 0; i < enNC; i++){
        enGMMCovar[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!enGMMCovar[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(enGMMCovar[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, encovarFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succesfully read the %s file.\n", encovarFile);
    fclose(fp);

    // for opening the prec file
    fp = fopen(enprecFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", enprecFile);
        exit(1);
    }
    enPrec  = (double**) malloc(sizeof(double*) * enNC);
    if (!enPrec){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", enNC);
        exit(1);
    }
    for (int i = 0; i < enNC; i++){
        enPrec[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!enPrec[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(enPrec[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, enprecFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n",enprecFile);
    fclose(fp);

    // for opening the sqrt precision file
    fp = fopen(ensqrtprecFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", ensqrtprecFile);
        exit(1);
    }
    enSqrtPrec  = (double**) malloc(sizeof(double*) * enNC);
    if (!enSqrtPrec){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", enNC);
        exit(1);
    }
    for (int i = 0; i < enNC; i++){
        enSqrtPrec[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!enSqrtPrec[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(enSqrtPrec[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, ensqrtprecFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n",ensqrtprecFile);
    fclose(fp);


    // for opening the enMeanPrecProd file
    fp = fopen(enmeanprecprodFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", enmeanprecprodFile);
        exit(1);
    }
    enMeanPrecProd  = (double**) malloc(sizeof(double*) * nFeatures);
    if (!enMeanPrecProd){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", nFeatures);
        exit(1);
    }
    for (int i = 0; i < nFeatures; i++){
        enMeanPrecProd[i] = (double*) malloc(sizeof(double) * enNC);
        if (!enMeanPrecProd[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", enNC);
            exit(1);
        }
        count = fread(enMeanPrecProd[i], sizeof(double), enNC, fp);
        if (count != enNC){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", enNC, i, enmeanprecprodFile);
            exit(1);
        }
    }
    fprintf(stderr,"Succefully read the %s file.\n", enmeanprecprodFile);
    fclose(fp);

}

void initFrenchGmmParameters(){
    FILE *fp;
    //for opening the weights
    fp = fopen(frweightFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", frweightFile);
        exit(1);
    }
    frGMMWeights = (double*) malloc(sizeof(double) * frNC);
    if (!frGMMWeights){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", frNC);
        exit(1);
    }
    int count = fread(frGMMWeights, sizeof(double), frNC, fp);
    if (count != frNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", frNC, frweightFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", frweightFile);
    }
    fclose(fp);

    //for opening the logweights
    fp = fopen(frlogweightFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", frlogweightFile);
        exit(1);
    }
    frLogWeight = (double*) malloc(sizeof(double) * frNC);
    if (!frLogWeight){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", frNC);
        exit(1);
    }
    count = fread(frLogWeight, sizeof(double), frNC, fp);
    if (count != frNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", frNC, frlogweightFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", frlogweightFile);
    }
    fclose(fp);

    //for opening the MeanPrecProdSum
    fp = fopen(frmeanprecprodsumFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", frmeanprecprodsumFile);
        exit(1);
    }
    frMeanPrecProdSum = (double*) malloc(sizeof(double) * frNC);
    if (!frMeanPrecProdSum){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", frNC);
        exit(1);
    }
    count = fread(frMeanPrecProdSum, sizeof(double), frNC, fp);
    if (count != frNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", frNC, frmeanprecprodsumFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", frmeanprecprodsumFile);
    }
    fclose(fp);

    //for opening the logDeterminant file frlogdetFile
    fp = fopen(frlogdetFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", frlogdetFile);
        exit(1);
    }
    frLogDet = (double*) malloc(sizeof(double) * frNC);
    if (!frLogDet){
        fprintf(stderr, "could not allocate %d number of double type memory blocks.\n", frNC);
        exit(1);
    }
    count = fread(frLogDet, sizeof(double), frNC, fp);
    if (count != frNC){
        fprintf(stderr, "could not read %d number of values from the file %s.\n", frNC, frlogdetFile);
        exit(1);
    } else {
        fprintf(stderr, "Succesfully read the file %s.\n", frlogdetFile);
    }
    fclose(fp);


    //for opening the mean file
    fp = fopen(frmeanFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", frmeanFile);
        exit(1);
    }
    frGMMMean  = (double**) malloc(sizeof(double*) * frNC);
    if (!frGMMMean){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", frNC);
        exit(1);
    }
    for (int i = 0; i < frNC; i++){
        frGMMMean[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!frGMMMean[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(frGMMMean[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, frmeanFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n",frmeanFile);
    fclose(fp);

    //for opening the covar file
    fp = fopen(frcovarFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", frcovarFile);
        exit(1);
    }
    frGMMCovar  = (double**) malloc(sizeof(double*) * frNC);
    if (!frGMMCovar){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", frNC);
        exit(1);
    }
    for (int i = 0; i < frNC; i++){
        frGMMCovar[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!frGMMCovar[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(frGMMCovar[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, frcovarFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", frcovarFile);
    fclose(fp);

    // for opening the prec file
    fp = fopen(frprecFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", frprecFile);
        exit(1);
    }
    frPrec  = (double**) malloc(sizeof(double*) * frNC);
    if (!frPrec){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", frNC);
        exit(1);
    }
    for (int i = 0; i < frNC; i++){
        frPrec[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!frPrec[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(frPrec[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, frprecFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", frprecFile);
    fclose(fp);

    // for opening the sqrt precision file
    fp = fopen(frsqrtprecFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", frsqrtprecFile);
        exit(1);
    }
    frSqrtPrec  = (double**) malloc(sizeof(double*) * frNC);
    if (!frSqrtPrec){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", frNC);
        exit(1);
    }
    for (int i = 0; i < frNC; i++){
        frSqrtPrec[i] = (double*) malloc(sizeof(double) * nFeatures);
        if (!frSqrtPrec[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", nFeatures);
            exit(1);
        }
        count = fread(frSqrtPrec[i], sizeof(double), nFeatures, fp);
        if (count != nFeatures){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", nFeatures, i, frsqrtprecFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", frsqrtprecFile);
    fclose(fp);


    // for opening the frMeanPrecProd file
    fp = fopen(frmeanprecprodFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing respective file.\n", frmeanprecprodFile);
        exit(1);
    }
    frMeanPrecProd  = (double**) malloc(sizeof(double*) * nFeatures);
    if (!frMeanPrecProd){
        fprintf(stderr, "Could not create %d number of double* type blocks.\n", nFeatures);
        exit(1);
    }
    for (int i = 0; i < nFeatures; i++){
        frMeanPrecProd[i] = (double*) malloc(sizeof(double) * frNC);
        if (!frMeanPrecProd[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", frNC);
            exit(1);
        }
        count = fread(frMeanPrecProd[i], sizeof(double), frNC, fp);
        if (count != frNC){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", frNC, i, frmeanprecprodFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", frmeanprecprodFile);
    fclose(fp);

}
void clearEnglishGmmParameters(){
    free(enGMMWeights);
    free(enLogDet);
    free(enLogWeight);
    free(enMeanPrecProdSum);

    for (int i = 0; i < enNC; i++){
        free(enGMMMean[i]);
        free(enGMMCovar[i]);
        free(enPrec[i]);
        free(enSqrtPrec[i]);
    }
    free(enGMMMean);
    free(enGMMCovar);
    free(enPrec);
    free(enSqrtPrec);
    for (int i = 0; i < nFeatures; i++){
        free(enMeanPrecProd[i]);
    }
    free(enMeanPrecProd);
    fprintf(stderr, "Succesfully removed various english gmm parameters from the memory blocks.\n");
}



void clearFrenchGmmParameters(){
    free(frGMMWeights);
    free(frLogWeight);
    free(frLogDet);
    free(frMeanPrecProdSum);

    for (int i = 0; i < frNC; i++){
        free(frGMMMean[i]);
        free(frGMMCovar[i]);
        free(frPrec[i]);
        free(frSqrtPrec[i]);
    }
    free(frGMMMean);
    free(frGMMCovar);
    free(frPrec);
    free(frSqrtPrec);
    for (int i = 0; i < nFeatures; i++){
        free(frMeanPrecProd[i]);
    }
    free(frMeanPrecProd);
    fprintf(stderr, "Succesfully removed various french gmm parameters from the memory blocks.\n");
}



void initEnglishTVmatrix(){
    FILE *fp;
    fp = fopen(enTVMatFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", enTVMatFile);
        exit(1);
    }
    int count;
    enTVMat = (double**) malloc(sizeof(double*) * nFeatures * enNC);
    for (int i = 0; i < nFeatures * enNC; i++){
        enTVMat[i] = (double*) malloc(sizeof(double) * enTVCol);
        if (!enTVMat[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", enTVCol);
            exit(1);
        }
        count = fread(enTVMat[i], sizeof(double), enTVCol, fp);
        if (count != enTVCol){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", enTVCol, i, enTVMatFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", enTVMatFile);
    fclose(fp);
}

void clearEnglishTVmatrix(){
    for (int i = 0; i < nFeatures * enNC; i++){
        free(enTVMat[i]);
    }
    free(enTVMat);
}


void initFrenchTVmatrix(){
    FILE *fp;
    fp = fopen(frTVMatFile,"r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", frTVMatFile);
        exit(1);
    }
    int count;
    frTVMat = (double**) malloc(sizeof(double*) * nFeatures * frNC);
    for (int i = 0; i < nFeatures * frNC; i++){
        frTVMat[i] = (double*) malloc(sizeof(double) * frTVCol);
        if (!frTVMat[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", frTVCol);
            exit(1);
        }
        count = fread(frTVMat[i], sizeof(double), frTVCol, fp);
        if (count != frTVCol){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", frTVCol, i, frTVMatFile);
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", frTVMatFile);
    fclose(fp);
}


void clearFrenchTVmatrix(){
    for (int i = 0; i < nFeatures * frNC; i++){
        free(frTVMat[i]);
    }
    free(frTVMat);
}

