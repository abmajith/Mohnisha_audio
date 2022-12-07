#include <stdio.h>
#include "frontend_functions.h"
#include "frontend_types.h"
#include <stdlib.h>


int main(){
    RawAudio rawAudio;
    rawAudio.deltaK = 3;
    rawAudio.sampleRate = 44100.0f;
    rawAudio.winLen = 0.02f;
    rawAudio.winStep = 0.01;
    rawAudio.lowFreq = 0.0f;
    rawAudio.highFreq = (double) rawAudio.sampleRate / 2.0f;
    rawAudio.preemph = 0.96f;
    rawAudio.numCept = 19;
    rawAudio.nFilt = 26;
    rawAudio.nFFT = 0;
    rawAudio.cepLifter = 22;
    rawAudio.len = 306176;
    rawAudio.audio = NULL;
    double *p = (double*) malloc(sizeof(double) * 306176);
    if (!p){
        exit(1);
    } else {
        rawAudio.audio= p;
    }

    
    FILE *fp;
    fp = fopen("testWav.bin","r");
    if (fp == NULL){
        fprintf(stderr, "testWav binary file can't opened now\n");
        exit(1);
    }
    int count = fread(rawAudio.audio, sizeof(double), rawAudio.len, fp);
    if (count != rawAudio.len){
        printf("Could not read %ld numbers.\n", (long) rawAudio.len);
        exit(1);
    } else {
        printf("Succesfully read the file.\n");
        fclose(fp);
        //printf("%lf, %lf\n",rawAudio.audio[0], rawAudio.audio[rawAudio.len-1]);
        //printf("%lf, %lf, %lf\n", rawAudio.audio[10], rawAudio.audio[100], rawAudio.audio[1000]);
    }
    
    MFCCFeatures Features;
    simpleMFCCExtraction(&rawAudio, &Features);
    //printMFCC(&Features);
    free(rawAudio.audio);
    return 1;
}
