#include <stdio.h>

int main(){
    int buf[3], out[3];
    for (int i = 0; i < 3; i++)
        buf[i] = out[i] = i;
    for (int i = 0; i < 3; i++){
        printf("%d\t%d\n",buf[i], out[i]);
    }
    return 1;
}
