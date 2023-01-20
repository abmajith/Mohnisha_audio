#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct RawAudio{
    int len;
    double *Audio;
} RawAudio;


void padAudio(RawAudio *rawAudio, int len){
    double *p = realloc(rawAudio->Audio, sizeof(double) * ((size_t) len + rawAudio->len) );
    if (!p){
        printf("Problem in reallocating the memory in padAudio function.\n");
        exit(1);
    } else {
        rawAudio->Audio = p;
    }
    printf("It came after reallocation of memory.\n");
    for (int i = rawAudio->len; i < rawAudio->len + len; i++)
        rawAudio->Audio[i] = 0.0f;
    rawAudio->len = rawAudio->len + len;
    printf("Size of rawAudio is %d\n",rawAudio->len);
}

void increase(RawAudio rawAudio, int len){
    printf("It come here \n");
    padAudio(&rawAudio, len);
    printf("This is success too\n");

}

int main(){
    RawAudio rawAudio;
    rawAudio.len = 3;
    rawAudio.Audio = (double*) malloc(sizeof(double) * 3);
    printf("Initializing rawAudio.\n");
    for (int i = 0; i < 3; i++){
        rawAudio.Audio[i] = (double) i+1.0f;
    }
    //printf("Upto this point is okay!\n");
    padAudio(&rawAudio, 2);
    printf("This is success.\n");
    increase(rawAudio, (int)2);
    for (int i = 0; i < 7; i++){
        printf(" %lf\t",rawAudio.Audio[i]);
    }
    printf("\n");
    return 1;
}
