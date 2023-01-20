#include <stdio.h>
#include <stdlib.h>


void extend(double *ty, int k){
    double *p = realloc(ty, ((size_t) sizeof(double) * (4 + k)));
    if (!p){
        printf("Failed to reallocate memeory.\n");
        exit(1);
    }
    if (p == ty){
        printf("Both are same.\n");
    }
}

int main(){
    double *t = (double*) malloc(sizeof(double) * 4);
    t[0] = 1.0;
    t[1] = 2.0;
    t[2] = 3.0;
    t[3] = 4.0;

    extend(t, 2);
    for (int i = 0; i < 6; i++){
        printf("%lf\t",t[i]);
    }
    printf("\n");
}
