#include "matrixoperation.h"
#include <stdio.h>

void initSqMat(SqMat *sqMat){
    if (sqMat->Row <= 0){
        printf("Given square matrix is having null dimension.\n");
        exit(1);
    }
    double **p = (double**) malloc(sizeof(double*) * sqMat->Row);
    if (!p){
        printf("Could not allocate %d number of double* type memory blocks.\n", sqMat->Row);
        exit(1);
    } else {
        sqMat->matrix = p;
    }
    double *q;
    for (int i = 0; i < sqMat->Row; i++){
        q = (double*) malloc(sizeof(double) * sqMat->Row);
        if (!q){
            printf("Could not allocate %d number of double type memory blocks.\n",sqMat->Row);
            exit(1);
        } else {
            sqMat->matrix[i] = q;
            for (int k = 0; k < sqMat->Row; k++){
                sqMat->matrix[i][k] = (double) 0.0f;
            }
        }
    }
}

void clearSqMat(SqMat *sqMat){
    for (int i = 0; i < sqMat->Row; i++){
        free(sqMat->matrix[i]);
    }
    free(sqMat->matrix);
}

void printMat(SqMat *sqMat){
    if (sqMat->Row <= 0){
        printf("Given square matrix is having null dimension.\n");
        exit(1);
    }
    printf("\n\n");
    for (int i = 0; i < sqMat->Row; i++){
        for (int j = 0; j < sqMat->Row; j++){
            printf(" %2.16lf\t",sqMat->matrix[i][j]);
        }
        printf("\n\n");
    }
}

void initIdentityMat(SqMat *Identity){
    if (Identity->Row <= 0){
        printf("Given square matrix is having null dimension.\n");
        exit(1);
    }
    double **p = (double**) malloc(sizeof(double*) * Identity->Row);
    if (!p){
        printf("Could not allocate %d number of double* type memory blocks.\n", Identity->Row);
        exit(1);
    } else {
        Identity->matrix = p;
    }
    double *q;
    for (int i = 0; i < Identity->Row; i++){
        q = (double*) malloc(sizeof(double) * Identity->Row);
        if (!q){
            printf("Could not allocate %d number of double type memory blocks.\n",Identity->Row);
            exit(1);
        } else {
            Identity->matrix[i] = q;
            for (int k = 0; k < Identity->Row; k++){
                Identity->matrix[i][k] = (double) 0.0f;
            }
            Identity->matrix[i][i] = (double) 1.0f;
        }
    }
}


void pivotSqMat(SqMat *sqMat, SqMat *zeros){
    int *p = (int*) malloc(sizeof(int) * sqMat->Row);
    if (!p){
        printf("Could not allocate %d number of int type memory blocks.\n", sqMat->Row);
        exit(1);
    } else {
        for (int i = 0; i < sqMat->Row; i++){
            p[i] = 0;
        }
    }
    double constant = 2.220446049250313e-15f;
    int maxId;
    double **q = (double**) malloc(sizeof(double*) * sqMat->Row);
    if (!q){
        printf("Could not allocate %d number of double* type memory blocks.\n",sqMat->Row);
        exit(1);
    }
    for (int r = 0; r < sqMat->Row; r++){
        maxId = 0;
        for (int c = 0; c < sqMat->Row; c++){
            if ( (abs(sqMat->matrix[r][c]) >= constant) && (p[maxId] == 0) ){
                p[maxId] = 1;
                zeros->matrix[maxId][r] = (double) 1.0f;
                q[r] = sqMat->matrix[maxId];
                break;
            }else {
                maxId += 1;
            }
        }
        //printf("(%d,%d)\t",r, maxId);
    }
    free(sqMat->matrix);
    sqMat->matrix = q;
    free(p);
    //printMat(sqMat);
}


void computeLUdecomposition(SqMat *sqMat, SqMat *L, SqMat *U){
    //printf("I am inside lu.\n");
    for (int i = 0; i < sqMat->Row; i++){
        for (int j = 0; j < sqMat->Row; j++){
            if (j >= i){
                U->matrix[i][j] = sqMat->matrix[i][j];
                for (int k = 0; k < i; k++){
                    U->matrix[i][j] -= L->matrix[i][k] * U->matrix[k][j];
                }
                if (i == j){
                    L->matrix[i][j] = 1.0f;
                } else {
                    L->matrix[i][j] = 0.0f;
                }
            } else {
                L->matrix[i][j] = sqMat->matrix[i][j];
                for (int k = 0; k < j; k++){
                    L->matrix[i][j] -= L->matrix[i][k] * U->matrix[k][j];
                }
                L->matrix[i][j] /= U->matrix[j][j];
                U->matrix[i][j] = 0.0f;
            }
        }
    }
}

void matMultiplication(SqMat *L, SqMat *U, SqMat *Res){
    for (int i = 0; i < L->Row; i++){
        for (int j = 0; j < L->Row; j++){
            Res->matrix[i][j] = 0.0f;
            for (int k = 0; k < L->Row; k++){
                Res->matrix[i][j] += L->matrix[i][k] * U->matrix[k][j];
            }
        }
    }
}

void invUppMatrixLU(SqMat *U, SqMat *Identity){
    double norm, scale;
    for (int i = U->Row - 1; i >= 0; i-- ){
        norm = U->matrix[i][i];
        for (int j = i; j < U->Row; j++){
            Identity->matrix[i][j] /= norm;
            scale = Identity->matrix[i][j];
            for (int k = i - 1; k >=0; k--){
                Identity->matrix[k][j] -= scale * U->matrix[k][i];
            }
        }
    }
}

void invLowMatrixLU(SqMat *L, SqMat *Identity){
    double scale;
    for (int i = 0; i < L->Row; i++){
        for (int j = 0; j <= i; j++){
            scale = Identity->matrix[i][j];
            for (int k = i + 1; k < L->Row; k++){
                Identity->matrix[k][j] -= scale * L->matrix[k][i];
            }
        }
    }
}


void inverseMatrixLU(SqMat *mat, SqMat *Res){
    SqMat La, Ua, ILa, IUa, Pa;
    La.Row = mat->Row;
    Ua.Row = mat->Row;
    Pa.Row = mat->Row;
    IUa.Row = mat->Row;
    ILa.Row = mat->Row;
    initSqMat(&La);
    initSqMat(&Ua);
    initIdentityMat(&ILa);
    initIdentityMat(&IUa);
    initSqMat(&Pa);
    //printf("I am going to print the given matrix.\n");
    //printMat(mat);

    //printf("I am going to compute pivot matrix.\n");
    pivotSqMat(mat, &Pa);
    //printf("I am going to print the given matrix.\n");
    //printMat(mat);
    //printf("I am going to compute lu decomposition.\n");
    computeLUdecomposition(mat, &La, &Ua);
    //printf("I just computed LU decomposition matrices.\n");

    //printf("goint to compute inverse upper matrix.\n");
    invUppMatrixLU(&Ua, &IUa);
    //printf("going to compute inverse lower matrix.\n");
    invLowMatrixLU(&La, &ILa);

    matMultiplication(&IUa, &ILa, &La);
    matMultiplication(&La, &Pa, Res);
    clearSqMat(&La);
    clearSqMat(&Ua);
    clearSqMat(&IUa);
    clearSqMat(&ILa);
    clearSqMat(&Pa);
}

void copyMatrix(SqMat *Src, SqMat *Des){
    Des->Row = Src->Row;
    double **p = (double**) malloc(sizeof(double*) * Src->Row);
    if (!p){
        printf("Could not allocate %d number of double* type memory blocks.\n", Src->Row);
        exit(1);
    } else {
        Des->matrix = p;
    }
    double *q;
    for (int i = 0; i < Src->Row; i++){
        q = (double*) malloc(sizeof(double) * Src->Row);
        if (!q){
            printf("Could not allocate %d number of double type memory blocks.\n",Src->Row);
            exit(1);
        } else {
            Des->matrix[i] = q;
            for (int j = 0; j < Src->Row; j++){
                Des->matrix[i][j] = Src->matrix[i][j];
            }
        }
    }
}


