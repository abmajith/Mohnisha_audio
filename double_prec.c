#include <stdio.h>

int main(){
    double x = 3.14567399278265f;
    float h = (float) x;
    double delta = x - ((double) h);
    float l = (float) delta;

    printf("%2.15lf, %2.15f, %2.15f\n",x, h, l);
    return 0;
}
