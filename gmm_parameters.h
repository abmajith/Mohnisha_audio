#ifndef GMM_PARAMETERS_H 
#define GMM_PARAMETERS_H
#include <stdlib.h>
#include "frontend_normalization_constants.h"



extern const int frNC;
extern const int enNC;


extern const char *frweightFile;
extern const char *frmeanFile;
extern const char *frcovarFile;
extern const char *frlogdetFile;
extern const char *frlogweightFile;
extern const char *frmeanprecprodFile;
extern const char *frmeanprecprodsumFile;
extern const char *frprecFile;
extern const char *frsqrtprecFile;

extern double *frGMMWeights;
extern double **frGMMMean;
extern double **frGMMCovar;
extern double *frLogDet;
extern double **frPrec;
extern double **frSqrtPrec;
extern double *frLogWeight;
extern double *frMeanPrecProdSum;
extern double **frMeanPrecProd;


extern const char *enweightFile;
extern const char *enmeanFile;
extern const char *encovarFile;
extern const char *enlogdetFile;
extern const char *enlogweightFile;
extern const char *enmeanprecprodFile;
extern const char *enmeanprecprodsumFile;
extern const char *enprecFile;
extern const char *ensqrtprecFile;

extern double *enGMMWeights;
extern double **enGMMMean;
extern double **enGMMCovar;
extern double *enLogDet;
extern double **enPrec;
extern double **enSqrtPrec;
extern double *enLogWeight;
extern double *enMeanPrecProdSum;
extern double **enMeanPrecProd;


//to load and clear the memory blocks to do some computation

void initEnglishGmmParameters();
void initFrenchGmmParameters();
void clearEnglishGmmParameters();
void clearFrenchGmmParameters();

#endif
