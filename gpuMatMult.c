#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>

typedef struct Mat{
    int nRow;
    int nCol;
    double *data;
} Mat;

void initMat(Mat *mat){
    double *p = (double*) malloc(sizeof(double) * mat->nRow * mat->nCol);
    if (!p){
        printf("Could not allocate %d number of double type memory blocks.\n", mat->nRow * mat->nCol);
        exit(1);
    } else {
        mat->data = p;
    }
}

void clearMat(Mat *mat){
    free(mat->data);
}


enum Consts {INFOLOG_LEN = 512};
GLchar infoLog[INFOLOG_LEN];
static const int GLSLShaderProgramLen = 1000;

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

void gpuMatMul(Mat *L, Mat *U, Mat *Res){
    if (L->nCol != U->nRow){
        printf("Matrices dimensions are not compatible to produce matrix multiplication.\n");
        exit(1);
    }
    if ((Res->nRow != L->nRow) || (Res->nCol != U->nCol)){
        printf("Res matrix dimension not matching with actual result of L * U.\n");
        exit(1);
    }
    
    //set the context for initializing the EGL and openGL stuffs here
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        exit(1);
    }
    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        exit(1);
    }
    //printf("EGL versions %u, %u\n", (unsigned int) majorVersion, (unsigned int) minorVersion);

    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_NONE };
    if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
        printf("eglChooseConfig failed\n");
        exit(1);
    }

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed\n");
        exit(1);
    }
    returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        exit(1);
    }
    //creating the compute shader
    GLint success;
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(1);
    }
    char *computerShaderProgram = (char*) malloc(sizeof(char) * GLSLShaderProgramLen );
    unsigned int lx, ly, M, N, K;//have to do this logic correct
    lx = 10u;
    ly = 10u;
    M = 100u;
    N = 100u;
    K = 100u;
    sprintf(computerShaderProgram, gComputeShaderMatMul, lx,ly,M,N,K);
    const char *ShaderCompute = computerShaderProgram;
    glShaderSource(computeShader, 1, &ShaderCompute, NULL);
    glCompileShader(computeShader);
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(computeShader, INFOLOG_LEN, NULL, infoLog);
        printf("ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n%s\n", infoLog);
    }
    //creating the shader program
    GLuint shaderProgram = glCreateProgram();
    if(!shaderProgram){
        printf("Failed to create a shader program.\n");
        exit(1);
    } else {
        glAttachShader(shaderProgram, computeShader);
        glLinkProgram(shaderProgram);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
        if (linkStatus == GL_FALSE){
            glGetProgramInfoLog(shaderProgram, INFOLOG_LEN, NULL, infoLog);
            fprintf(stderr, "Could not link program:\n%s\n", infoLog);
        }
    }
    //arranging the matrix as arrays
    GLuint matASSbo;
    GLuint matBSSbo;
    GLuint matCSSbo;
    GLuint sizeA, sizeB, sizeC;
    sizeA = (GLuint) L->nRow * L->nCol;
    sizeB = (GLuint) U->nRow * U->nCol;
    sizeC = (GLuint) Res->nRow * Res->nCol;

    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeA * sizeof(double), L->data, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeB * sizeof(double), U->data, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matCSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeC * sizeof(double), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matCSSbo);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    glUseProgram(shaderProgram);
    glDispatchCompute(10,10,1);//have to do this logic correct
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    double* pOut = (double*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeC * sizeof(double), GL_MAP_READ_BIT);
    for (int i = 0; i < sizeC; i++){
        Res->data[i] = pOut[i];
    }

    eglDestroyContext(dpy, context);
    eglTerminate(dpy);
}

int main(){
    Mat L, U, Res;
    L.nRow = L.nCol = U.nRow = U.nCol = Res.nRow = Res.nCol = 100;
    initMat(&L);
    initMat(&U);
    initMat(&Res);
    for (int i = 0; i < 10000; i++){
        L.data[i] = U.data[i] = (double) i/10;
    }
    gpuMatMul(&L, &U, &Res);
    for (int i = 0; i < 100; i++){
        printf("%.10lf\t",Res.data[i]);
    }
    return 0;
}
