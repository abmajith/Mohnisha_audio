#ifndef MATRIXOPERATION_H 
#define MATRIXOPERATION_H
#include <stdlib.h>
#include <math.h>

typedef struct SqMat{
    int Row;
    double **matrix;
} SqMat;

void initSqMat(SqMat *sqMat);
void clearSqMat(SqMat *sqMat);
void printMat(SqMat *sqMat);
void initIdentityMat(SqMat *Identity);
void pivotSqMat(SqMat *sqMat, SqMat *zeros);
void computeLUdecomposition(SqMat *sqMat, SqMat *L, SqMat *U);
void matMultiplication(SqMat *L, SqMat *U, SqMat *Res);
void invUppMatrixLU(SqMat *U, SqMat *Identity);
void invLowMatrixLU(SqMat *L, SqMat *Identity);
void inverseMatrixLU(SqMat *mat, SqMat *Res);
void copyMatrix(SqMat *Src, SqMat *Des);
#endif
