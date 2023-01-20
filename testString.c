#include <stdio.h>
#include <math.h>
#include <string.h>


char *gComputeShaderMatMul =
    "#version 320 es\n"
    "layout(local_size_x = %u, local_size_y = %u) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "   double data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "   double data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "   double data[];\n"
    "} matC;\n"
    "shared uint M = %uu;\n"
    "shared uint N = %uu;\n"
    "shared uint K = %uu;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   double acc = double(0.0);\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k*M + globalRow] * matB.data[globalCol * K + k];\n"
    "matC.data[globalRow + globalCol * M] = acc;\n"
    "}\n";


int main(){
    int l = strlen(gComputeShaderMatMul);
    printf("%d\n",l);
    unsigned int M,N,K,lx, ly;
    lx = 10u;
    ly = 10u;
    M = 100u;
    N = 100u;
    K = 100u;
    char dst[l+10];
    sprintf(dst, gComputeShaderMatMul,lx,ly, M,N,K);
    printf("%s\n",dst);
    l = strlen(dst);
    printf("%d\n",l);

    return 0;
}
