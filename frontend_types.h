#ifndef FRONTEND_TYPES_H
#define FRONTEND_TYPES_H

#include <math.h>
#include <stdlib.h>

//this is really a stupid way of writing the structure. Almost all struct looks similar, but keep as of it is now. will do later some improvement
//to store raw audio signal
typedef struct RawAudio{
    double *audio;
    int len;
    double sampleRate;
    double winLen;
    double winStep;
    double lowFreq;
    double highFreq;
    double preemph;
    int numCept;
    int nFilt;
    int nFFT;
    int cepLifter;
    int deltaK;
} RawAudio;

//to store framed audio signal
typedef struct SignalFrames {
    int numberOfFrames;
    int frameLength;
    int nextFrameIndex;
    double sampleRate;
    double winLen;
    double winStep;
    double **frames;
} SignalFrames;

typedef struct FramedEnergySpectrum{
    int numberOfFrames;
    int numberOfFilters;
    double **energy;
}FramedEnergySpectrum;

//to store the number of filterbanks
typedef struct  FilterBanks {
    int numberOfFilters;
    int numberOfFreq;
    int nFFT;
    double sampleRate;
    double highFreq;
    double lowFreq;
    double **fbanks;
} FilterBanks;

//to store the MFCC Features
typedef struct MFCCFeatures {
    int numberOfFrames;
    int numberOfCepstrum;
    double **Features;
} MFCCFeatures;

typedef struct FrameFreqSpectrum{
    int numberOfFrames;
    int nFFT;
    int numberOfPositiveFrequency; // (int) floor(nFFT / 2) + 1
    double **framedfrequencySpectrum;
} FrameFreqSpectrum;

//allocate appropriate memory for defined structures
void initFramedEnergySpectrumMem(FramedEnergySpectrum *energy);
void initFrameFreqSpectrum(FrameFreqSpectrum *frameFreqSpectrum);
void initSignalFramesMem(SignalFrames *sigFrame);
void initFilterBanksMem(FilterBanks *filterBank);
void initMFCCFeaturesMem(MFCCFeatures *mfccFeature);




//delete respective structures
void delFramedEnergySpectrumMem(FramedEnergySpectrum *energy);
void delFrameFreqSpectrum(FrameFreqSpectrum *frameFreqSpectrum);
void delSignalFramesMem(SignalFrames *sigFrame);
void delFilterBanksMem(FilterBanks *filterBank);
void delMFCCFeaturesMem(MFCCFeatures *mfccFeature);

#endif
