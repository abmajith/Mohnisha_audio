#include <stdio.h>
#include "frontend_types.h"


void initFramedEnergySpectrumMem(FramedEnergySpectrum *energy){
    double **p = (double**) malloc(sizeof(double*) * energy->numberOfFrames);
    if (!p){
        printf("Memory could not allocate in initFramedEnergySpectrumMem function.\n");
        exit(1);
    } else {
        energy->energy = p;
    }
    for (int n_f = 0; n_f < energy->numberOfFrames; n_f++){
        double *p = (double*) malloc(sizeof(double) * energy->numberOfFilters);
        if (!p){
        printf("Memory could not allocate in initFramedEnergySpectrumMem function.\n");
        exit(1);
        }else {
           energy->energy[n_f] = p;
        }
    }
}

void delFramedEnergySpectrumMem(FramedEnergySpectrum *energy){
    for (int n_f = 0; n_f < energy->numberOfFrames; n_f++){
        free(energy->energy[n_f]);
    }
    free(energy->energy);
}

void initFrameFreqSpectrum(FrameFreqSpectrum *frameFreqSpectrum){
    double **p = (double**) malloc(sizeof(double*) * frameFreqSpectrum->numberOfFrames);
    if (!p){
        printf("Memory could not allocate in initFrameFreqSpectrum function.\n");
        exit(1);
    } else {
        frameFreqSpectrum->framedfrequencySpectrum = p;
    }
    for (int n_f = 0; n_f < frameFreqSpectrum->numberOfFrames; n_f++){
        double *p = (double*) malloc(sizeof(double) * frameFreqSpectrum->numberOfPositiveFrequency);
        if (!p){
            printf("Memory could not allocate in initFrameFreqSpectrum function.\n");
            exit(1);
        }else {
            frameFreqSpectrum->framedfrequencySpectrum[n_f] = p;
        }
    }
}

void delFrameFreqSpectrum(FrameFreqSpectrum *frameFreqSpectrum){
    for (int n_f = 0; n_f < frameFreqSpectrum->numberOfFrames; n_f++){
        free(frameFreqSpectrum->framedfrequencySpectrum[n_f]);
    }
    free(frameFreqSpectrum->framedfrequencySpectrum);
}

void initSignalFramesMem(SignalFrames *sigFrame){
    double **p = (double**) malloc(sizeof(double*) * sigFrame->numberOfFrames);
    if (!p){
        printf("Memory could not allocate in initSignalFramesMem function.\n");
        exit(1);
    }else {
        sigFrame->frames = p;
    }
    for (int n_f = 0; n_f < sigFrame->numberOfFrames; n_f++){
        double *p= (double*) malloc(sizeof(double) * sigFrame->frameLength);
        if (!p){
            printf("Memory could not allocate in initSignalFramesMem function.\n");
            exit(1);
        }else{
            sigFrame->frames[n_f] = p;
        }
    }
}

void delSignalFramesMem(SignalFrames *sigFrame){
    for (int n_f = 0; n_f < sigFrame->numberOfFrames; n_f++){
        free(sigFrame->frames[n_f]);
    }
    free(sigFrame->frames);
}


void initFilterBanksMem(FilterBanks *filterBank){
    double **p = (double**) malloc(sizeof(double*) * filterBank->numberOfFilters);
    if (!p){
        printf("Memory could not allocate in initFilterBanksMem function.\n");
        exit(1);
    }else{
            filterBank->fbanks = p;
    }
    for (int filt = 0; filt < filterBank->numberOfFilters; filt++){
        double *p = (double*) malloc(sizeof(double) * filterBank->numberOfFreq);
        if (!p){
            printf("Memory could not allocate in initFilterBanksMem function.\n");
            exit(1);
        }else{
            filterBank->fbanks[filt] = p;
            for (int i = 0; i < filterBank->numberOfFreq; i++)
                filterBank->fbanks[filt][i] = (double) 0.0f;
        }
    }
}

void delFilterBanksMem(FilterBanks *filterBank){
    for (int filt = 0; filt < filterBank->numberOfFilters; filt++){
        free(filterBank->fbanks[filt]);
    }
    free(filterBank->fbanks);
}

void initMFCCFeaturesMem(MFCCFeatures *mfccFeature){
    double **p = (double**) malloc(sizeof(double*) * mfccFeature->numberOfFrames);
    if (!p){
        printf("Memory could not allocate in initMFCCFeaturesMem function.\n");
        exit(1);
    } else {
        mfccFeature->Features = p;
    }
    for (int n_f = 0; n_f < mfccFeature->numberOfFrames; n_f++){
        double *p = (double*) malloc(sizeof(double) * 2 * mfccFeature->numberOfCepstrum);
        if (!p){
        printf("Memory could not allocate in initMFCCFeaturesMem function.\n");
        exit(1);
        }else {
           mfccFeature->Features[n_f] = p;
        }
    }
}

void delMFCCFeaturesMem(MFCCFeatures *mfccFeature){
    for (int n_f = 0; n_f < mfccFeature->numberOfFrames; n_f++){
        free(mfccFeature->Features[n_f]);
    }
    free(mfccFeature->Features);
}

