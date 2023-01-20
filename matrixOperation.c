#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Mat{
    int nRow;
    int nCol;
    double *data;
} Mat;

void initMat(Mat *mat){
    double *p = (double*) malloc(sizeof(double) * mat->nRow * mat->nCol);
    if (!p){
        printf("Could not allocate %d number of double type memory blocks.\n", mat->nRow * mat->nCol);
        exit(1);
    } else {
        mat->data = p;
    }
}

void clearMat(Mat *mat){
    free(mat->data);
}

void printMat(Mat *mat){
    printf("\n\n");
    /*
    double* p[mat->nRow];
    for (int i = 0; i < mat->nRow; i++){
        p[i] = mat->data + (i * mat->nCol);
    }
    */
    for (int i = 0; i < mat->nRow; i++){
        for (int j = 0; j < mat->nCol; j++){
            printf(" %2.16lf\t",mat->data[i * mat->nCol + j]);
            //printf(" %2.16lf\t",*(p[i] + j));
        }
        printf("\n\n");
    }
}

void initIdentityMat(Mat *identity){
    /*
    if (identity->nCol != identity->nRow){
        printf("Given matrix is not square matrix (%d by %d) is the size of given matrix.\n",identity->nRow, identity->nCol);
        exit(1);
    }
    */
    double *p = (double*) malloc(sizeof(double) * identity->nRow * identity->nCol);
    if (!p){
        printf("Could not allocate %d number of double type memory blocks.\n", identity->nRow * identity->nCol);
        exit(1);
    } else {
        identity->data = p;
    }
    int nrow = identity->nRow;
    int ncol = identity->nCol;
    for (int i = 0; i < nrow; i++){
        for (int j = 0; j < ncol; j++){
            if (i != j)
                identity->data[i * ncol + j] = 0.0f;
            else 
                identity->data[i * ncol + j] = 1.0f;
        }
    }
}

void pivotSqMat(Mat *sqMat, Mat *zeros){
    if ((sqMat->nRow != sqMat->nCol) || (zeros->nCol != zeros->nRow)){
        printf("Given matrix are not sqmatrix.\n");
        exit(1);
    }
    if ((sqMat->nCol != zeros->nCol) || (sqMat->nRow != zeros->nRow)){
        printf("Given two matrices are not same dimensions.\n");
        exit(1);
    }
    int *p = (int*) malloc(sizeof(int) * sqMat->nRow);
    if (!p){
        printf("Could not allocate %d number of int type memory blocks.\n", sqMat->nRow);
        exit(1);
    } else {
        for (int i = 0; i < sqMat->nRow; i++)
            p[i] = 0;
    }
    int nrow = sqMat->nRow;
    int ncol = sqMat->nCol;
    double constant = 2.220446049250313e-15f;
    
    double *q = (double*) malloc(sizeof(double) * nrow * ncol);
    if (!q){
        printf("Could not allocate %d number of double type memory blocks.\n", nrow * ncol);
        exit(1);
    }
    
    int r = 0;
    for (int c = 0; c < ncol; c++){
        for (int maxId = 0; maxId < nrow; maxId++){
            if ( (abs(sqMat->data[maxId * ncol + c]) > constant) && (p[maxId] == 0) ){
                printf("(%d , %d)\t",r,maxId);
                p[maxId] = 1;
                zeros->data[maxId * ncol + c] = (double) 1.0f;
                for (int j = 0; j < ncol; j++)
                    q[r * ncol + j] = sqMat->data[maxId * ncol + j];
                r += 1;
                break;
            }
        }
    }
    printf("\n\n");
    free(sqMat->data);
    sqMat->data = q;
    free(p);
}


void computeLUdecomposition(Mat *sqMat, Mat *L, Mat *U){
    if ((sqMat->nRow != sqMat->nCol) || (L->nRow != L->nCol) || (U->nRow != U->nCol)){
        printf("One of the given matrix are not square matrix.\n");
        exit(1);
    }
    if ((sqMat->nRow != L->nRow) || (sqMat->nRow != U->nRow)){
        printf("Given matrices are not having the same dimensions.\n");
        exit(1);
    }
    int nrow = sqMat->nRow;
    int ncol = sqMat->nCol;
    for (int i = 0; i < nrow; i++){
        for (int j = 0; j < ncol; j++){
            if (j >= i){
                U->data[i * ncol + j] = sqMat->data[i * ncol + j];
                for (int k = 0; k < i; k++){
                    U->data[i * ncol + j] -= L->data[i * ncol + k] * U->data[k * ncol + j];
                }
                if (i == j){
                    L->data[i * ncol + j] = 1.0f;
                } else {
                    L->data[i * ncol + j] = 0.0f;
                }
            } else {
                L->data[i * ncol + j] = sqMat->data[i * ncol + j];
                for (int k = 0; k < j; k++){
                    L->data[i * ncol + j] -= L->data[i * ncol + k] * U->data[k * ncol + j];
                }
                L->data[i * ncol + j] /= U->data[j * ncol + j];
                U->data[i * ncol + j] = 0.0f;
            }
        }
    }
}

void matMultiplication(Mat *L, Mat *U, Mat *Res){
    if (L->nCol != U->nRow){
        printf("Matrices dimensions are not compatible to produce matrix multiplication.\n");
        exit(1);
    }
    if ((Res->nRow != L->nRow) || (Res->nCol != U->nCol)){
        printf("Res matrix dimension not matching with actual result of L * U.\n");
        exit(1);
    }
    int nrow = L->nRow;
    int ncol = U->nCol;
    int K = L->nCol;
    for (int i = 0; i < nrow; i++){
        for (int j = 0; j < ncol; j++)
            Res->data[i * ncol + j] = 0.0f;
        for (int k = 0; k < K; k++){
            for (int j = 0; j < ncol; j++)
                Res->data[i * ncol + j] += L->data[i * K + k] * U->data[k * ncol + j];
        }
    }
}

void invUppMatrixLU(Mat *U, Mat *identity){
    if ((U->nRow != U->nCol) || (identity->nRow != identity->nCol)){
        printf("Given matrices are not square.\n");
        exit(1);
    }
    if (U->nRow != identity->nRow){
        printf("Given matrices are not having the same dimensions.\n");
        exit(1);
    }
    int nrow = U->nRow;
    int ncol = U->nCol;
    double norm, scale;
    for (int i = nrow - 1; i >= 0; i-- ){
        norm = U->data[i * ncol + i];
        for (int j = i; j < nrow; j++){
            identity->data[i * ncol + j] /= norm;
            scale = identity->data[i * ncol + j];
            for (int k = i - 1; k >=0; k--){
                identity->data[k * ncol + j] -= scale * U->data[k * ncol + i];
            }
        }
    }
}

void invLowMatrixLU(Mat *L, Mat *identity){
    if ((L->nRow != L->nCol) || (identity->nRow != identity->nCol)){
        printf("Given matrices are not square.\n");
        exit(1);
    }
    if (L->nRow != identity->nRow){
        printf("Given matrices are not having the same dimensions.\n");
        exit(1);
    }
    int nrow = L->nRow;
    int ncol = L->nCol;
    double scale;
    for (int i = 0; i < nrow; i++){
        for (int j = 0; j <= i; j++){
            scale = identity->data[i * ncol + j];
            for (int k = i + 1; k < nrow; k++){
                identity->data[k * ncol + j] -= scale * L->data[k * ncol + i];
            }
        }
    }
}

void inverseMatrixLU(Mat *mat, Mat *Res){
    Mat La, Ua, ILa, IUa, Pa;
    La.nRow = mat->nRow;
    La.nCol = mat->nCol;
    Ua.nRow = mat->nRow;
    Ua.nCol = mat->nCol;
    Pa.nRow = mat->nRow;
    Pa.nCol = mat->nCol;
    IUa.nRow = mat->nRow;
    IUa.nCol = mat->nCol;
    ILa.nRow = mat->nRow;
    ILa.nCol = mat->nCol;
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
    Des->nRow = Src->nRow;
    Des->nCol = Src->nCol;
    double *p = (double*) malloc(sizeof(double) * Src->nRow * Src->nCol);
    if (!p){
        printf("Could not allocate %d number of double type memory blocks.\n", Src->nRow * Src->nCol);
        exit(1);
    } else {
        Des->data = p;
        for (int i = 0; i < Src->nRow * Src->nCol; i++)
            Des->data[i] = Src->data[i];
    }
}

int main(void){
    int dim = 4;
    Mat SqMat;
    SqMat.nRow = dim;
    SqMat.nCol = dim;
    initMat(&SqMat);
    //double Row1[] = {1.0f, 0.0f, 2.0f};
    //double Row2[] = {2.0f, -1.0f, 3.0f};
    //double Row3[] = {4.0f, 1.0f, 8.0f};
    double Row[] = {0.0f, 2.0f, 4.0f, 1.0f, 2.0f, 1.0f, 0.1f, 0.0f, 3.0f};
    double row[] = {11.0f,9.0f, 24.0f, 2.0f, 1.0f,5.0f,2.0f,6.0f, 3.0f,17.0f,18.0f,1.0f, 2.0f,5.0f,7.0f,1.0f};
    for (int i = 0; i < 16; i++){
        SqMat.data[i] = row[i];
    }
    printf("Original Matrix.\n");
    printMat(&SqMat);
    
    Mat L, U;
    L.nRow = dim;
    U.nRow = dim;
    L.nCol = dim;
    U.nCol = dim;
    Mat pivotMatrix;
    pivotMatrix.nRow = dim;
    pivotMatrix.nCol = dim;
    initMat(&pivotMatrix);
    pivotSqMat(&SqMat, &pivotMatrix);
    //printf("pivot matrix .\n");
    //printMat(&pivotMatrix);
    //printf(" and the respective shuffeled matrix is \n");
    //printMat(&SqMat);




    initMat(&L);
    initMat(&U);
    computeLUdecomposition(&SqMat, &L, &U);
    Mat Res;
    Res.nRow = dim;
    Res.nCol = dim;
    initMat(&Res);
    Mat IdentityU;
    IdentityU.nRow = dim;
    IdentityU.nCol = dim;
    Mat IdentityL;
    IdentityL.nRow = dim;
    IdentityL.nCol = dim;
    initIdentityMat(&IdentityL);
    initIdentityMat(&IdentityU);


    invUppMatrixLU(&U, &IdentityU);
    invLowMatrixLU(&L, &IdentityL);
    //printf("U matrix.\n");
    //printMat(&U);
    //printf("Inverse of U matrix.\n");
    //printMat(&IdentityU);
    matMultiplication(&U, &IdentityU, &Res);
    //printf("Result of U U-1.\n");
    //printMat(&Res);

    //printf("L matrix.\n");
    //printMat(&L);
    //printf("Inverse of L matrix.\n");
    //printMat(&IdentityL);
    matMultiplication(&L, &IdentityL, &Res);
    //printf("Result of L L-1.\n");
    //printMat(&Res);



    matMultiplication(&IdentityU, &IdentityL, &Res);
    matMultiplication(&Res, &pivotMatrix, &L);
    matMultiplication(&SqMat, &L, &Res);
    //printf("After finding inverse of a matrix using LU decomposition, we check that SqMat * inverse SqMat result as : \n");
    //printMat(&Res);

    clearMat(&SqMat);
    clearMat(&pivotMatrix);
    clearMat(&L);
    clearMat(&U);
    clearMat(&Res);

    
    Mat sqMat;
    sqMat.nRow = 3;
    sqMat.nCol = 3;
    Mat res;
    res.nRow = 3;
    res.nCol = 3;
    initMat(&res);
    initMat(&sqMat);
    for (int i = 0; i < 9; i++){
        sqMat.data[i] = Row[i];
    }
    
    Mat cMat;
    cMat.nRow = 3;
    cMat.nCol = 3;
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
