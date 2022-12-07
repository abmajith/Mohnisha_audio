#include "matrixoperation.h"

void initSqMat(SqMat *sqMat){
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
    printf("\n\n");
    for (int i = 0; i < sqMat->Row; i++){
        for (int j = 0; j < sqMat->Row; j++){
            printf(" %2.16lf\t",sqMat->matrix[i][j]);
        }
        printf("\n\n");
    }
}

void initIdentityMat(SqMat *Identity){
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
    }
    free(sqMat->matrix);
    sqMat->matrix = q;
    free(p);
}

void computeLUdecomposition(SqMat *sqMat, SqMat *L, SqMat *U){
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
    pivotSqMat(mat, &Pa);
    computeLUdecomposition(mat, &La, &Ua);
    invUppMatrixLU(&Ua, &IUa);
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


