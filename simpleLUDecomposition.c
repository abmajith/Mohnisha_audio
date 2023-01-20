#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Mat{
    int Row;
    double **matrix;
} Mat;

void initMat(Mat *sqMat){
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

void clearMat(Mat *sqMat){
    for (int i = 0; i < sqMat->Row; i++){
        free(sqMat->matrix[i]);
    }
    free(sqMat->matrix);
}

void printMat(Mat *sqMat){
    printf("\n\n");
    for (int i = 0; i < sqMat->Row; i++){
        for (int j = 0; j < sqMat->Row; j++){
            printf(" %2.16lf\t",sqMat->matrix[i][j]);
        }
        printf("\n\n");
    }
}

void initIdentityMat(Mat *Identity){
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


void pivotSqMat(Mat *sqMat, Mat *zeros){
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


void computeLUdecomposition(Mat *sqMat, Mat *L, Mat *U){

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

void matMultiplication(Mat *L, Mat *U, Mat *Res){
    for (int i = 0; i < L->Row; i++){
        for (int j = 0; j < L->Row; j++){
            Res->matrix[i][j] = 0.0f;
            for (int k = 0; k < L->Row; k++){
                Res->matrix[i][j] += L->matrix[i][k] * U->matrix[k][j];
            }
        }
    }
}

void invUppMatrixLU(Mat *U, Mat *Identity){
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

void invLowMatrixLU(Mat *L, Mat *Identity){
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

void inverseMatrixLU(Mat *mat, Mat *Res){
    Mat La, Ua, ILa, IUa, Pa;
    La.Row = mat->Row;
    Ua.Row = mat->Row;
    Pa.Row = mat->Row;
    IUa.Row = mat->Row;
    ILa.Row = mat->Row;
    initMat(&La);
    initMat(&Ua);
    initIdentityMat(&ILa);
    initIdentityMat(&IUa);
    initMat(&Pa);
    pivotSqMat(mat, &Pa);
    computeLUdecomposition(mat, &La, &Ua);
    invUppMatrixLU(&Ua, &IUa);
    invLowMatrixLU(&La, &ILa);
    matMultiplication(&IUa, &ILa, &La);
    matMultiplication(&La, &Pa, Res);
    clearMat(&La);
    clearMat(&Ua);
    clearMat(&IUa);
    clearMat(&ILa);
    clearMat(&Pa);
}

void copyMatrix(Mat *Src, Mat *Des){
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

int main(void){
    int dim = 4;
    Mat SqMat;
    SqMat.Row = dim;
    initMat(&SqMat);
    //double Row1[] = {1.0f, 0.0f, 2.0f};
    //double Row2[] = {2.0f, -1.0f, 3.0f};
    //double Row3[] = {4.0f, 1.0f, 8.0f};
    double Row1[] = {0.0f, 2.0f, 4.0f};
    double Row2[] = {1.0f, 2.0f, 1.0f};
    double Row3[] = {0.1f, 0.0f, 3.0f};
    double row1[] = {11.0f,9.0f, 24.0f, 2.0f};
    double row2[] = {1.0f,5.0f,2.0f,6.0f};
    double row3[] = {3.0f,17.0f,18.0f,1.0f};
    double row4[] = {2.0f,5.0f,7.0f,1.0f};
    for (int i = 0; i < dim; i++){
        SqMat.matrix[0][i] = row1[i];
        SqMat.matrix[1][i] = row2[i];
        SqMat.matrix[2][i] = row3[i];
        SqMat.matrix[3][i] = row4[i];
    }
    printf("Original Matrix.\n");
    printMat(&SqMat);
    Mat L, U;
    L.Row = dim;
    U.Row = dim;
    Mat pivotMatrix;
    pivotMatrix.Row = dim;
    initMat(&pivotMatrix);
    pivotSqMat(&SqMat, &pivotMatrix);
    printf("pivot matrix .\n");
    printMat(&pivotMatrix);
    printf(" and the respective shuffeled matrix is \n");
    printMat(&SqMat);




    initMat(&L);
    initMat(&U);
    computeLUdecomposition(&SqMat, &L, &U);
    Mat Res;
    Res.Row = dim;
    initMat(&Res);
    Mat IdentityU;
    IdentityU.Row = dim;
    Mat IdentityL;
    IdentityL.Row = dim;
    initIdentityMat(&IdentityL);
    initIdentityMat(&IdentityU);


    invUppMatrixLU(&U, &IdentityU);
    invLowMatrixLU(&L, &IdentityL);
    printf("U matrix.\n");
    printMat(&U);
    printf("Inverse of U matrix.\n");
    printMat(&IdentityU);
    matMultiplication(&U, &IdentityU, &Res);
    printf("Result of U U-1.\n");
    printMat(&Res);

    printf("L matrix.\n");
    printMat(&L);
    printf("Inverse of L matrix.\n");
    printMat(&IdentityL);
    matMultiplication(&L, &IdentityL, &Res);
    printf("Result of L L-1.\n");
    printMat(&Res);



    matMultiplication(&IdentityU, &IdentityL, &Res);
    matMultiplication(&Res, &pivotMatrix, &L);
    matMultiplication(&SqMat, &L, &Res);
    printf("After finding inverse of a matrix using LU decomposition, we check that SqMat * inverse SqMat result as : \n");
    printMat(&Res);

    clearMat(&SqMat);
    clearMat(&pivotMatrix);
    clearMat(&L);
    clearMat(&U);
    clearMat(&Res);


    Mat sqMat;
    sqMat.Row = 3;
    Mat res;
    res.Row = 3;
    initMat(&res);
    initMat(&sqMat);
    for (int i = 0; i < 3; i++){
        sqMat.matrix[0][i] = Row1[i];
        sqMat.matrix[1][i] = Row2[i];
        sqMat.matrix[2][i] = Row3[i];
    }
    Mat cMat;
    copyMatrix(&sqMat, &cMat);
    printMat(&cMat);
    inverseMatrixLU(&sqMat, &res);
    printMat(&res);
    matMultiplication(&res, &cMat, &sqMat);
    printMat(&sqMat);
    clearMat(&res);
    clearMat(&sqMat);
    clearMat(&cMat);
    return 1;
}
