#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <complex.h>

void _rfft(double *A, int nFFT, int lenA, double *R, double *I){
    int N = nFFT;
    double twoPI = 2.0f * 3.14159265358979323846f;
    double *paddedA = NULL;
    if (nFFT < lenA){
        printf("Consider to increase nFFT\n");
        paddedA = A;
    } else {
        paddedA = (double*) malloc(sizeof(double) * nFFT);
        for (int i = 0; i < lenA; i++)
            paddedA[i] = A[i];
        for (int i = lenA; i < nFFT; i++)
            paddedA[i] = 0.0f;
    }
    int n = floor(nFFT / 2) + 1;
    R = (double*) malloc(sizeof(double) * n);
    I = (double*) malloc(sizeof(double) * n);
    double theta, angle;
    for (int k = 0; k < n; k++){
        R[k] = 0.0;
        I[k] = 0.0;
        printf(" %lf + (%lf)j \t",R[k], I[k]);
        theta = (double) twoPI *  ((double) k / N);
        for (int m = 0; m < N; m++){
            angle = theta *  (double)  m;
            R[k] += paddedA[m] * cos(angle);
            I[k] -= paddedA[m] * sin(angle);
        }
         printf("\t\t %lf + (%lf)j\n",R[k], I[k]);
    }
    printf("\n\n");
    /*
    for (int i = 0; i < n; i++){
        printf(" %lf + (%lf)j \t",R[i], I[i]);
    }
    printf("\n\n");
    */
}

int main(){
    double A[] = {1,1,1,1,1,1,1,1,1};
    double *I = NULL;
    double *R = NULL;
    _rfft(A, 8, 8, I, R);
}
