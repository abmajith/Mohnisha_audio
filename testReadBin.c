#include <stdlib.h>
#include <stdio.h>


int main(){
    FILE *fp;
    fp = fopen("enWeights.bin","r");
    if (fp == NULL){
        fprintf(stderr, "frenchGMMWeights can't opened now\n");
        exit(1);
    }
    double *weights = NULL;
    weights = (double*) malloc(sizeof(double) * 1019);
    int count = fread(weights, sizeof(double), 1019, fp);
    fclose(fp);
    printf("Elements read: %d\n", count);
    double sum = 0.0f;
    for (int i = 0; i < 1019; i++){
        printf("%2.16lf \t", weights[i]);
        sum += weights[i];
    }
    printf("\n\n");
    printf("\n %10.10lf\n", sum);

    double **Means = NULL;
    Means = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        Means[i] = (double*) malloc(sizeof(double) * 38);
    }
    fp = fopen("enMean.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "frenchMeans can't opened now\n");
        exit(1);
    }
    for (int i = 0; i < 1019; i++){
        fread(Means[i], sizeof(double), 38, fp);
    }
    fclose(fp);
    for (int i = 0; i < 2; i++){
        printf("M[%d]\n",i);
        for (int d = 0; d < 38; d++){
            printf("%10.20lf\t",Means[i][d]);
        }
        printf("\n\n");
    }
    printf("\n\nM[]\n");
    for (int d = 0; d < 38; d++){
            printf("%10.20lf\t",Means[1018][d]);
    }
    printf("\n\n");
    fp = fopen("enCovar.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "frenchCovar can't opened now\n");
        exit(1);
    }
    double **Covar = NULL;
    Covar = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        Covar[i] = (double*) malloc(sizeof(double) * 38);
    }

    for (int i = 0; i < 1019; i++){
        fread(Covar[i], sizeof(double), 38, fp);
    }
    fclose(fp);

    for (int i = 0; i < 2; i++){
        printf("C[%d]\n",i);
        for (int d = 0; d < 38; d++){
            printf("%10.20lf\t",Covar[i][d]);
        }
        printf("\n\nC[1018]\n");
    }
    for (int d = 0; d < 38; d++){
            printf("%10.20lf\t",Covar[1018][d]);
    }
    printf("\n\n");


    return 0;
}
