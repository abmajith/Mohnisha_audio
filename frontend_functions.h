#ifndef FRONTEND_FUNCTIONS_H 
#define FRONTEND_FUNCTIONS_H 

#include "frontend_types.h"
#include "frontend_normalization_constants.h"

void padAudioSignal(RawAudio *rawAudio, int padLen);
void padFramedSignal(SignalFrames *framedSignal, int padLen);

void preEmphasis(RawAudio *rawAudio, double coeff);
void findFrames(RawAudio *rawAudio, SignalFrames *framedSignal);
void windowingFrames(SignalFrames *framedSignal);


void rfft(double rbuf[], double ibuf[], double rout[], double iout[], int nFFT, int step);
int CalculateNFFt(double sampleRate, double winLen);
double Mel2Hz(double mel);
double Hz2Mel(double hz);
void LinSpace(double lowP, double highP, int nP, double* Points);

void computeFilterBanks(FilterBanks *fbanks);

void ApplyDFTSpectrum(SignalFrames *framedSignal, FrameFreqSpectrum *freqSpectrum);
void EnergySpectrum(FrameFreqSpectrum *frameFreqSpectrum, FilterBanks *filterBanks, FramedEnergySpectrum *logEnergy);
void DCTtypeTwo(FramedEnergySpectrum *logEnergy, MFCCFeatures *Features);
void Lifter(MFCCFeatures *Features, int lifter);

void computeDeltaFeatures(MFCCFeatures *Features, int K);

void cmnvNormFeatures(MFCCFeatures *Features, const double *mean, const double *std, const int dim);

void simpleMFCCExtraction(RawAudio *rawAudio, MFCCFeatures *Features);

void printMFCC(MFCCFeatures *Features);

#endif
