#include "frontend_functions.h"
#include <stdio.h>
#include <time.h>

//verified
void padAudioSignal(RawAudio *rawAudio, int padLen){
    double *p = realloc(rawAudio->audio, ((size_t) sizeof(double) * (rawAudio->len + padLen)));
    if (!p){
        printf("%d number of double type Memory could not allocate in padAudioSignal function\n",padLen);
        exit(1);
    } else {
        rawAudio->audio = p;
    }
    for (int i = rawAudio->len; i < rawAudio->len + padLen; i++){
        rawAudio->audio[i] = (double) 0.0f;
    }
    // increase respectively the length of the audio len indicator
    rawAudio->len += padLen;
}

//verified
void padFramedSignal(SignalFrames *framedSignal, int padLen){
    for (int n_f = 0; n_f < framedSignal->numberOfFrames; n_f++){
        double *p = realloc(framedSignal->frames[n_f], ((size_t) sizeof(double) * (padLen + framedSignal->frameLength)));
        if (!p){
            printf("%d number of double type Memory could not allocate in padFramedSignal function\n",padLen);
            exit(1);
        } else {
            framedSignal->frames[n_f] = p;
        }
        for (int i = framedSignal->frameLength; i < padLen + framedSignal->frameLength; i++){
            framedSignal->frames[n_f][i] = (double) 0.0f;
        }
    }
    framedSignal->frameLength += padLen;
}

//verified
void preEmphasis(RawAudio *rawAudio, double coeff){
    clock_t start, end;
    double execution_time;
    start = clock();
    double lastValue = rawAudio->audio[0];
    double nextLastValue;
    for (int i = 1; i < rawAudio->len; i++){
        nextLastValue = rawAudio->audio[i];
        rawAudio->audio[i] -= coeff * lastValue;
        lastValue = nextLastValue;
    }
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do preemphasis.\n",execution_time);
}

//verified
void findFrames(RawAudio *rawAudio, SignalFrames *framedSignal){
    clock_t start, end;
    double execution_time;
    start = clock();   
    int frameLength = framedSignal->frameLength;
    int nextFrameIndex = framedSignal->nextFrameIndex;
    int numberOfFrame = framedSignal->numberOfFrames;

    for (int n_f = 0; n_f < numberOfFrame; n_f++){
        for (int k = 0; k < frameLength; k++){
            framedSignal->frames[n_f][k] = rawAudio->audio[k + (n_f * nextFrameIndex)];
        }
    }
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to find Frames.\n",execution_time);
}

//verified
void windowingFrames(SignalFrames *framedSignal){
    clock_t start, end;
    double execution_time;
    start = clock();
    double twoPI = 2.0f * 3.14159265358979323846f;
    double window;
    int frameLength = framedSignal->frameLength;
    for (int i = 0; i < framedSignal->frameLength; i++){
        window = (double) 0.54f - 0.46f * cos(twoPI * ((double) i / (frameLength - 1)));
        for (int n_f = 0; n_f < framedSignal->numberOfFrames; n_f++){
            framedSignal->frames[n_f][i] *= window;
        }
    }
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do windowingFrames.\n",execution_time);
}

//verified
void rfft(double rbuf[], double ibuf[], double rout[], double iout[], int nFFT, int step){
    if (step < nFFT){
        rfft(rout, iout, rbuf, ibuf, nFFT, step * 2);
        rfft(rout + step, iout + step, rbuf + step, ibuf + step, nFFT, step * 2);
        double rt, it, cosine, sine;
        double piconst = 3.14159265358979323846f;
        double angle;
        for (int i = 0; i < nFFT; i += 2 * step){
            angle = piconst * ((double) i / nFFT);
            cosine = cos(angle);
            sine = sin(angle);
            rt = cosine * rout[i + step] + sine * iout[i + step];
            it = iout[i + step] * cosine - rout[i + step] * sine;
            rbuf[i / 2] = rout[i] + rt;
            ibuf[i / 2] = iout[i] + it;
            rbuf[(i + nFFT) / 2] = rout[i] - rt;
            ibuf[(i + nFFT) / 2] = iout[i] - it;
        }
    }
}

//verified
int CalculateNFFt(double sampleRate, double winLen){
    int windowLengthSample = (int) floor((double) sampleRate * winLen);
    int nFFT = 1;
    for (int i = 0; i < windowLengthSample; i++){
        nFFT *= 2;
        if (nFFT >= windowLengthSample)
            return nFFT;
    }
    return nFFT;
}

//verified
double Mel2Hz(double mel){
    double melScale = (double) mel / 2595.0f;
    double hertz = (double) 700.0f * (pow( (double) 10.0f, (double) melScale) - 1.0f);
    return hertz;
}

//verified
double Hz2Mel(double hz){
    double hzScale = 1.0f + (double) hz / 700;
    return (double) 2595.0f * (double) log10((double) hzScale);
}

//verified
void LinSpace(double lowP, double highP, int nP, double *Points){
    int N = nP - 1;
    double incValue = (double) (highP - lowP) / N;
    Points[0] = lowP;
    Points[N] = highP;
    for (int i = 1; i < N; i++){
        Points[i] = Points[i-1] + incValue;
    }
}

//verified
void computeFilterBanks(FilterBanks *fbanks){
    int nFFT = fbanks->nFFT;
    double lowFreq = fbanks->lowFreq;
    double highFreq = fbanks->highFreq;
    double sampleRate = fbanks->sampleRate;
    if (highFreq == 0.0){
        highFreq = (double) sampleRate / 2.0f;
        fbanks->highFreq = highFreq;
    }
    double lowMel = Hz2Mel(lowFreq);
    double highMel = Hz2Mel(highFreq);
    int nfilt = fbanks->numberOfFilters;
    double melPoints[nfilt + 2];
    LinSpace(lowMel, highMel, nfilt+2, melPoints);
    int bins[nfilt+2];
    for (int i = 0; i < nfilt + 2; i++){
        bins[i] = (int) floor( ((double) (nFFT + 1)) * Mel2Hz(melPoints[i]) / sampleRate);
    }
    for (int j = 0; j < nfilt; j++){
        for (int i = bins[j]; i < bins[j+1]; i++){
            fbanks->fbanks[j][i] = ((double)  i - bins[j]) / ((double)  bins[j+1] - bins[j]);
        }
        for (int i = bins[j+1]; i < bins[j+2]; i++){
            fbanks->fbanks[j][i] = ((double) bins[j+2]-i) / ((double) bins[j+2] - bins[j+1]);
        }
    }
}

//verified, commented algorithm is the substitute of fft process, but consume lot of time
void ApplyDFTSpectrum(SignalFrames *framedSignal, FrameFreqSpectrum *freqSpectrum){
    int nFFT = freqSpectrum->nFFT;
    int n = freqSpectrum->numberOfPositiveFrequency;
    double *rbuf = (double*) malloc(sizeof(double) * nFFT);
    double *ibuf = (double*) malloc(sizeof(double) * nFFT);
    double *rout = (double*) malloc(sizeof(double) * nFFT);
    double *iout = (double*) malloc(sizeof(double) * nFFT);
    if ((!rbuf) || (!ibuf) || (!rout) || (!iout)){
        printf("Could not allocate %d number of double type memory blocks.\n",nFFT);
        exit(1);
    }
    clock_t start, end;
    double execution_time;
    start = clock();
    for (int n_f = 0; n_f < framedSignal->numberOfFrames; n_f++){
        for (int i = 0; i < nFFT; i++){
            rbuf[i] = rout[i] = framedSignal->frames[n_f][i];
            ibuf[i] = iout[i] = 0.0f;
        }
        rfft(rbuf, ibuf, rout, iout, nFFT, (int) 1);
        for (int k = 0; k < n; k++){
            freqSpectrum->framedfrequencySpectrum[n_f][k] = (rbuf[k] * rbuf[k]) + (ibuf[k] * ibuf[k]);
        }
    }
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do all the rfft.\n",execution_time);
    free(rbuf);
    free(ibuf);
    free(rout);
    free(iout);
}

//verified
void EnergySpectrum(FrameFreqSpectrum *frameFreqSpectrum, FilterBanks *filterBanks, FramedEnergySpectrum *logEnergy){
    long double intRes;
    double smallConst = log((double) 2.220446049250313e-16);
    for (int n_f = 0; n_f < frameFreqSpectrum->numberOfFrames; n_f++){
        for (int filtN = 0; filtN < filterBanks->numberOfFilters; filtN++){
            intRes = (long double) 0.0f;
            for (int freq = 0; freq < filterBanks->numberOfFreq; freq++){
                intRes += (long double) frameFreqSpectrum->framedfrequencySpectrum[n_f][freq]  *  filterBanks->fbanks[filtN][freq];
            }
            if (intRes == 0.0f){
                logEnergy->energy[n_f][filtN] = smallConst;
            }else {
                logEnergy->energy[n_f][filtN] = (double) logl(intRes);
                
            }
        }
    }
}

//verified
void DCTtypeTwo(FramedEnergySpectrum *logEnergy, MFCCFeatures *Features){
    double norm = sqrt(logEnergy->numberOfFilters);
    double intRes;
    //storing zeroth order log energy spectrum of all frames
    for (int n_f = 0; n_f < logEnergy->numberOfFrames; n_f++){
        intRes = 0.0f;
        for (int filtN = 0; filtN < logEnergy->numberOfFilters; filtN++)
            intRes += logEnergy->energy[n_f][filtN];
        Features->Features[n_f][0] = intRes / norm;
    }
    //storing higher order log energy discrete cosine transformation of all frames
    norm = sqrt(((double) logEnergy->numberOfFilters / 2.0));
    double multPi, cosine, constMultiPi;
    constMultiPi = (double) 3.14159265358979323846f / ((double) 2.0f * (logEnergy->numberOfFilters));
    for (int n_f = 0; n_f < logEnergy->numberOfFrames; n_f++){
        for (int n_cept = 1; n_cept < Features->numberOfCepstrum; n_cept++ ){
            multPi = (double) constMultiPi * ((double) n_cept);
            intRes = 0.0f;
            for (int filtN = 0; filtN < logEnergy->numberOfFilters; filtN++){
                cosine = cos( multPi * (double) (2 * filtN + 1));
                intRes += logEnergy->energy[n_f][filtN] * cosine;
            }
            Features->Features[n_f][n_cept] = intRes / norm;
        }
    }
}

//verified
void Lifter(MFCCFeatures *Features, int lifter){
    if (lifter > 0){
        double normPi = (double) 3.14159265358979323846f / (double) lifter;
        double multiplier;
        double halfLifter = (double) lifter / 2.0f;
        for (int ceptN = 0; ceptN < Features->numberOfCepstrum; ceptN++){
            multiplier = 1.0f + halfLifter * sin(((double) ceptN * normPi));
            for (int n_f = 0; n_f < Features->numberOfFrames; n_f++){
                Features->Features[n_f][ceptN] *= multiplier;
            }
        }
    }
}

//verifed
void computeDeltaFeatures(MFCCFeatures *Features, int K){
    //note this function only works when you have more than 2*K number of frames to comute the delta of MFFC features
    if (Features->numberOfFrames < (2 * K + 1)){
        printf("\n\nNumber of frames is less than twice the number of delta function average, \n either decrease K or provide increased number of Frames\n");
        exit(1);
    }
    double norm = 3.0f /((double) K * (K + 1) * (2 * K + 1));
    int numCept = Features->numberOfCepstrum;
    double *firstRow = (double*) malloc(sizeof(double) * numCept);
    if (!firstRow){
        printf("Memory could not allocate in computeDeltaFeatures");
        exit(1);
    }
    double *lastRow = (double*) malloc(sizeof(double) * numCept);
    if (!lastRow){
        printf("Memory could not allocate in computeDeltaFeatures");
        exit(1);
    }
    for (int i = 0; i < Features->numberOfCepstrum; i++){
        firstRow[i] = Features->Features[0][i];
        lastRow[i] = Features->Features[Features->numberOfFrames - 1][i];
    }
    double intRes;
    int numberofframes = Features->numberOfFrames;
    for (int n_f = 0 ; (n_f < K) && ( (n_f + K) < numberofframes); n_f++){
        for (int f_f = numCept; f_f < 2 * numCept; f_f++){
            intRes = 0.0f;
            for (int k = K; k > 0; k--){
                if ( (n_f - k) < 0){
                    intRes -= ((double) k * firstRow[f_f - numCept]);
                } else {
                    intRes -= ((double) k * Features->Features[n_f - k][f_f - numCept]);
                }
                intRes += ((double) k * Features->Features[n_f + k][f_f - numCept]);
            }
            Features->Features[n_f][f_f] = (double) norm * intRes;
        }
    }
    for (int n_f = numberofframes - 1; (n_f > (numberofframes - K - 1)) && ((n_f - K) >= 0); n_f--){
        for (int f_f = numCept; f_f < 2 * numCept; f_f++){
            intRes = 0.0f;
            for (int k = K; k > 0; k--){
                if ((n_f + k) > (numberofframes - 1)){
                    intRes += ((double) k * lastRow[f_f - numCept]);
                } else {
                    intRes += ((double) k * Features->Features[n_f + k][f_f - numCept]);
                }
                intRes -= ((double) k * Features->Features[n_f - k][f_f - numCept]);
            }
            Features->Features[n_f][f_f] = (double) norm * intRes;
        }
    }
    for (int n_f = K; n_f < (numberofframes - K); n_f++){
        for (int f_f = numCept; f_f < 2 * numCept; f_f++){
            intRes = 0.0f;
            for (int k = K; k > 0; k--){
                intRes += ((double) k * Features->Features[n_f + k][f_f - numCept]);
                intRes -= ((double) k * Features->Features[n_f - k][f_f - numCept]);
            }
            Features->Features[n_f][f_f] = (double) norm * intRes;
        }
    }
}


void cmnvNormFeatures(MFCCFeatures *Features, const double *mean, const double *std, const int dim){
    if ((2 * (Features->numberOfCepstrum))  != dim){
        printf("The number of features are not same as given dim (%d).\n\n", dim);
        exit(1);
    }
   for (int d = 0; d < dim; d++){
       for (int n_f = 0; n_f < Features->numberOfFrames; n_f++){
           Features->Features[n_f][d] -= mean[d];
           Features->Features[n_f][d] /= std[d];
       }
   }
}

//verified
void simpleMFCCExtraction(RawAudio *rawAudio, MFCCFeatures *Features){
    if (rawAudio->nFFT == 0){
        rawAudio->nFFT = CalculateNFFt((double) rawAudio->sampleRate, (double) rawAudio->winLen);
    }
    if (rawAudio->highFreq == 0.0f){
        rawAudio->highFreq = rawAudio->sampleRate / 2.0f;
    }
    if ((rawAudio->preemph < 0.0f) || (rawAudio->preemph > 0.0f)){
        preEmphasis(rawAudio, rawAudio->preemph);
    }


    double winLen = rawAudio->winLen;
    double sampleRate = rawAudio->sampleRate;
    double winStep = rawAudio->winStep;

    if (winLen <= 0.0){
        printf("Given winLen (%lf) is not positive.\n\n", winLen);
        exit(1);
    }
    int frameLength = (int) floor((double) winLen * sampleRate);
    if (frameLength > rawAudio->len){
        printf("frameLength is greater than audio Signal Length.\n\n");
        exit(1);
    }
    int nextFrameIndex = (int) floor((double) winStep * sampleRate);
    int numberOfFrame = ((int) ceil(rawAudio->len - frameLength + nextFrameIndex) / nextFrameIndex);
    if (rawAudio->len < nextFrameIndex){
        printf("frameLength is shorter than hopFrameLength.\n\n");
        exit(1);
    }
    int reqSpeechLength = (numberOfFrame - 1) * nextFrameIndex + frameLength;
    if (rawAudio->len < reqSpeechLength){
        padAudioSignal(rawAudio, ((int) reqSpeechLength - rawAudio->len));
    }

    SignalFrames framedSignal;
    framedSignal.frameLength = frameLength;
    framedSignal.numberOfFrames = numberOfFrame;
    framedSignal.nextFrameIndex = nextFrameIndex;
    framedSignal.sampleRate = sampleRate;
    framedSignal.winLen = winLen;
    framedSignal.winStep = winStep;
    initSignalFramesMem(&framedSignal);

    findFrames(rawAudio, &framedSignal);
    windowingFrames(&framedSignal);
    //upto this point it is computing as expected
    //there is a problem from the next line to the next blue colored texts

    FrameFreqSpectrum freqSpectrum;
    freqSpectrum.numberOfFrames = framedSignal.numberOfFrames;
    freqSpectrum.nFFT = rawAudio->nFFT;
    freqSpectrum.numberOfPositiveFrequency = (int) floor(rawAudio->nFFT / 2) + 1;
    initFrameFreqSpectrum(&freqSpectrum);
    if (freqSpectrum.nFFT <= framedSignal.frameLength){
        printf("WARNINGS: number of FFT is lesser than frameLength, will loss some input audio information.\n");
    } else {
        //int padLen = ((int) freqSpectrum.nFFT - framedSignal.frameLength);
        padFramedSignal(&framedSignal,((int) freqSpectrum.nFFT - framedSignal.frameLength));
    }

    ApplyDFTSpectrum(&framedSignal, &freqSpectrum);
    FilterBanks fbanks;
    fbanks.numberOfFilters = rawAudio->nFilt;
    fbanks.numberOfFreq = freqSpectrum.numberOfPositiveFrequency;
    fbanks.nFFT = rawAudio->nFFT;
    fbanks.sampleRate = rawAudio->sampleRate;
    fbanks.highFreq = rawAudio->highFreq;
    fbanks.lowFreq = rawAudio->lowFreq;
    initFilterBanksMem(&fbanks);
    
    computeFilterBanks(&fbanks);
    
    FramedEnergySpectrum logEnergy;
    logEnergy.numberOfFrames = framedSignal.numberOfFrames;
    logEnergy.numberOfFilters = rawAudio->nFilt;
    initFramedEnergySpectrumMem(&logEnergy);

    EnergySpectrum(&freqSpectrum, &fbanks, &logEnergy);


    Features->numberOfFrames = framedSignal.numberOfFrames;
    Features->numberOfCepstrum = rawAudio->numCept;
    initMFCCFeaturesMem(Features);

    DCTtypeTwo(&logEnergy, Features);
    if (rawAudio->cepLifter > 0){
        Lifter(Features, (int) rawAudio->cepLifter);
    }
    if (rawAudio->deltaK > 0){
        computeDeltaFeatures(Features, (int) rawAudio->deltaK);
    }
    

    delFramedEnergySpectrumMem(&logEnergy);
    delFrameFreqSpectrum(&freqSpectrum);
    delSignalFramesMem(&framedSignal);
    delFilterBanksMem(&fbanks);

}

//verified
void printMFCC(MFCCFeatures *Features){
    printf("\nPrinting mfcc features that contains %d number of frames with %d number of features.\n",Features->numberOfFrames, Features->numberOfCepstrum);
    for (int nf = 0; nf < Features->numberOfFrames; nf++){
        printf("\n");
        for (int f = 0; f < 2 * Features->numberOfCepstrum; f++){
            printf("%lf\t",Features->Features[nf][f]);
        }
    }
    printf("\n\n");
}
