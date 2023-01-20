#include <stdlib.h>
#include <stdio.h>


int main(){
    FILE *outFile = fopen("FORANDROIDGMM.c","w");
    if (!outFile){
        printf("Could not open file.\n");
        exit(1);
    }

    fprintf(outFile, "#include \"gmm_parameters.h\" \n#include <stdlib.h> \nconst int enNC = 1019; \nconst int enTVCol = 150; \n");

    fprintf(outFile, "const double enGMMWeights[] = {");

    FILE *fp;
    fp = fopen("enWeights.bin","r");
    if (fp == NULL){
        fprintf(stderr, "englishGMMWeights can't open now\n");
        exit(1);
    }
    double *weights = NULL;
    weights = (double*) malloc(sizeof(double) * 1019);
    int count = fread(weights, sizeof(double), 1019, fp);
    fclose(fp);
    if (count < 1019){
        fprintf(stderr, "Could not read 1019 number of elements.\n");
        exit(1);
    }
    for (int i = 0; i < 1018; i++){
            fprintf(outFile, "%5.30lff, ", weights[i]);
    }
    fprintf(outFile, "%5.30lff", weights[1018]);
    free(weights);

    fprintf(outFile, "};\n\n");

    fprintf(outFile, "const double enLogDet[] = {");
    
    fp = fopen("enLogDet.bin","r");
    if (fp == NULL){
        fprintf(stderr, "englishLogDet can't open now\n");
        exit(1);
    }
    double *logdet = NULL;
    logdet = (double*) malloc(sizeof(double) * 1019);
    count = fread(logdet, sizeof(double), 1019, fp);
    fclose(fp);
    if (count < 1019){
        fprintf(stderr, "Could not read 1019 number of elements.\n");
        exit(1);
    }
    for (int i = 0; i < 1018; i++){
            fprintf(outFile, "%7.30lff, ", logdet[i]);
    }
    fprintf(outFile, "%7.30lff", logdet[1018]);
    free(logdet);

    fprintf(outFile, "};\n\n");
    

    fprintf(outFile, "const double enLogWeight[] = {");
    
    fp = fopen("enLogWeights.bin","r");
    if (fp == NULL){
        fprintf(stderr, "englishLogWeights can't open now\n");
        exit(1);
    }
    double *logweight = NULL;
    logweight = (double*) malloc(sizeof(double) * 1019);
    count = fread(logweight, sizeof(double), 1019, fp);
    fclose(fp);
    if (count < 1019){
        fprintf(stderr, "Could not read 1019 number of elements.\n");
        exit(1);
    }
    for (int i = 0; i < 1018; i++){
            fprintf(outFile, "%9.30lff, ", logweight[i]);
    }
    fprintf(outFile, "%9.30lff", logweight[1018]);
    free(logweight);

    fprintf(outFile, "};\n\n");

    fprintf(outFile, "const double enMeanPrecProdSum[] = {");

    fp = fopen("enMeanPrecProdSum.bin","r");
    if (fp == NULL){
        fprintf(stderr, "englishMeanPrecProdSum can't open now\n");
        exit(1);
    }
    double *meanprecprodsum = NULL;
    meanprecprodsum = (double*) malloc(sizeof(double) * 1019);
    count = fread(meanprecprodsum, sizeof(double), 1019, fp);
    fclose(fp);
    if (count < 1019){
        fprintf(stderr, "Could not read 1019 number of elements.\n");
        exit(1);
    }

    for (int i = 0; i < 1018; i++){
        fprintf(outFile, "%9.30lff, ", meanprecprodsum[i]);
    }
    fprintf(outFile, "%9.30lff",meanprecprodsum[1018]);
    free(meanprecprodsum);

    fprintf(outFile, "};\n\n");
    

    fprintf(outFile, "const double enGMMMean[enNC][] = { ");

    double **Means = NULL;
    Means = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        Means[i] = (double*) malloc(sizeof(double) * 38);
    }
    fp = fopen("enMean.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "EnglishMeans can't opened now\n");
        exit(1);
    }
    for (int i = 0; i < 1019; i++){
        fread(Means[i], sizeof(double), 38, fp);
    }
    fclose(fp);
    for (int i = 0; i < 1018; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 37; j++){
            fprintf(outFile, "%9.30lff, ", Means[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", Means[i][37]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 37; j++){
        fprintf(outFile, "%9.30lff, ", Means[1018][j]);
    }
    fprintf(outFile, "%9.30lff}",Means[1018][37]);
    for (int i = 0; i < 1019; i++){
        free(Means[i]);
    }
    free(Means);
    fprintf(outFile, "};\n\n");
    
    fprintf(outFile, "const double enGMMCovar[enNC][] = { ");

    double **Covar = NULL;
    Covar = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        Covar[i] = (double*) malloc(sizeof(double) * 38);
    }
    fp = fopen("enCovar.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "EnglishCovar can't opened now\n");
        exit(1);
    }
    for (int i = 0; i < 1019; i++){
        fread(Covar[i], sizeof(double), 38, fp);
    }
    fclose(fp);
    for (int i = 0; i < 1018; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 37; j++){
            fprintf(outFile, "%9.30lff, ", Covar[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", Covar[i][37]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 37; j++){
        fprintf(outFile, "%9.30lff, ", Covar[1018][j]);
    }
    fprintf(outFile, "%9.30lff}",Covar[1018][37]);
    for (int i = 0; i < 1019; i++){
        free(Covar[i]);
    }
    free(Covar);
    fprintf(outFile, "};\n\n");
    

    fprintf(outFile, "const double enPrec[enNC][] = { ");

    double **Prec = NULL;
    Prec = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        Prec[i] = (double*) malloc(sizeof(double) * 38);
    }
    fp = fopen("enPrec.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "EnglishPrec can't opened now\n");
        exit(1);    
    }
    for (int i = 0; i < 1019; i++){
        fread(Prec[i], sizeof(double), 38, fp);
    }
    fclose(fp);
    for (int i = 0; i < 1018; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 37; j++){
            fprintf(outFile, "%9.30lff, ", Prec[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", Prec[i][37]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 37; j++){
        fprintf(outFile, "%9.30lff, ", Prec[1018][j]);
    }
    fprintf(outFile, "%9.30lff}",Prec[1018][37]);
    for (int i = 0; i < 1019; i++){
        free(Prec[i]);
    }
    free(Prec);
    fprintf(outFile, "};\n\n");
    
    

    fprintf(outFile, "const double enSqrtPrec[enNC][] = { ");

    double **SqrtPrec = NULL;
    SqrtPrec = (double**) malloc(sizeof(double*) * 1019);
    for (int i = 0; i < 1019; i++){
        SqrtPrec[i] = (double*) malloc(sizeof(double) * 38);
    }
    fp = fopen("enSqrtPrec.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "EnglishSqrtPrec can't opened now\n");
        exit(1);
    }
    for (int i = 0; i < 1019; i++){
        fread(SqrtPrec[i], sizeof(double), 38, fp);
    }
    fclose(fp);
    for (int i = 0; i < 1018; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 37; j++){
            fprintf(outFile, "%9.30lff, ", SqrtPrec[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", SqrtPrec[i][37]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 37; j++){
        fprintf(outFile, "%9.30lff, ", SqrtPrec[1018][j]);
    }
    fprintf(outFile, "%9.30lff}",SqrtPrec[1018][37]);
    for (int i = 0; i < 1019; i++){
        free(SqrtPrec[i]);
    }
    free(SqrtPrec);
    fprintf(outFile, "};\n\n");


    fprintf(outFile, "const double enMeanPrecProd[nFeatures][] = { ");

    double **MeanPrecProd = NULL;
    MeanPrecProd = (double**) malloc(sizeof(double*) * 38);
    for (int i = 0; i < 38; i++){
        MeanPrecProd[i] = (double*) malloc(sizeof(double) * 1019);
    }
    fp = fopen("enMeanPrecProd.bin", "r");
    if (fp == NULL){
        fprintf(stderr, "EnglishMeanPrecProd can't opened now\n");
        exit(1);
    }
    for (int i = 0; i < 38; i++){
        fread(MeanPrecProd[i], sizeof(double), 1019, fp);
    }
    fclose(fp);
    for (int i = 0; i < 37; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 1018; j++){
            fprintf(outFile, "%9.30lff, ", MeanPrecProd[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", MeanPrecProd[i][1018]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 1018; j++){
        fprintf(outFile, "%9.30lff, ", MeanPrecProd[37][j]);
    }
    fprintf(outFile, "%9.30lff}",MeanPrecProd[37][1018]);
    for (int i = 0; i < 38; i++){
        free(MeanPrecProd[i]);
    }
    free(MeanPrecProd);
    fprintf(outFile, "};\n\n");


    fprintf(outFile, "const double enTVMat[nFeatures * enNC][] = { ");

    fp = fopen("englishTVmat.bin","r");
    if (fp == NULL){
        fprintf(stderr, "%s can't open now, missing the respective file.\n", "englishTVmat.bin");
        exit(1);
    }
    double **TVMat = NULL;
    TVMat = (double**) malloc(sizeof(double*) * 38 * 1019);
    for (int i = 0; i < 38 * 1019; i++){
        TVMat[i] = (double*) malloc(sizeof(double) * 150);
        if (!TVMat[i]){
            fprintf(stderr, "could not create %d number of double type memory block.\n", 150);
            exit(1);
        }
        count = fread(TVMat[i], sizeof(double), 150, fp);
        if (count != 150){
            fprintf(stderr, "could not read %d number of values at the index %d from the file %s.\n", 150, i, "englishTVmat.bin");
            exit(1);
        }
    }
    fprintf(stderr, "Succefully read the %s file.\n", "englishTVmat.bin");
    fclose(fp);
    

    for (int i = 0; i < 38 * 1019 - 1; i++){
        fprintf(outFile, "{");
        for (int j = 0; j < 149; j++){
            fprintf(outFile, "%9.30lff, ", TVMat[i][j]);
        }
        fprintf(outFile, "%9.30lff}, \n", TVMat[i][149]);
    }
    fprintf(outFile, "{");
    for (int j = 0; j < 149; j++){
        fprintf(outFile, "%9.30lff, ", TVMat[38 * 1019 - 1][j]);
    }
    fprintf(outFile, "%9.30lff}", TVMat[38 * 1019 - 1][149]);
    for (int i = 0; i < 38 * 1019; i++){
        free(TVMat[i]);
    }
    free(TVMat);
    fprintf(outFile, "};\n\n");

    fclose(outFile);
    return 0;
}
