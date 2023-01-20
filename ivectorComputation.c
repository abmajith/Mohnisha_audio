#include "ivectorComputation.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>



static const char *gComputeShaderIntRes =
    "#version 320 es\n"
    "layout(local_size_x = 150, local_size_y = 2) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "   double data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "   double data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "   double data[];\n"
    "} matC;\n"
    "shared uint M = 150u;\n"
    "shared uint N = 38722u;\n"
    "shared uint K = 150u;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   double acc = double(0.0);\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k*M + globalRow] * matB.data[globalCol * K + k];\n"
    "matC.data[globalRow + globalCol * M] = acc;\n"
    "}\n";


static const char *gComputeShaderIVector =
    "#version 320 es\n"
    "layout(local_size_x = 150,local_size_y = 2) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "    double data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "    double data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "    double data[];\n"
    "} matC;\n"
    "shared uint M = 150u;\n"
    "shared uint N = 1u;\n"
    "shared uint K = 38722u;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   double acc = double(0.0);\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k*M + globalRow] * matB.data[globalCol * K + k];\n"
    "matC.data[globalRow + globalCol * M] = acc;\n"
    "}\n";

static const char *gComputeShaderPosteriorPrecMatrix = 
    "#version 320 es\n"
    "layout(local_size_x = 30,local_size_y = 30) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "    double data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "    double data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "    double data[];\n"
    "} matC;\n"
    "shared uint nCol = 150u;\n"
    "shared uint NF = 38u;\n"
    "shared uint NC = 1019u;\n"
    "shared uint nRow = 38722u;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   double acc = double(0.0);\n"
    "   uint index = 0u;\n"
    "   double result = double(0.0);\n"
    "   if (globalRow == globalCol){\n"
    "       result = double(1.0);\n"
    "   }\n"
    "   for (uint nc = 0u; nc < NC; nc++){\n"
    "       acc = double(0.0);\n"
    "       index = nc * NF;\n"
    "       for (uint nf = 0u; nf < NF; nf++){\n"
    "           acc += double(matA.data[(globalRow * nRow) + index + nf] * matA.data[(globalCol * nRow) + index + nf]);\n"
    "       }\n"
    "       result += double(acc * matB.data[nc]);\n"
    "   }\n"
    "   matC.data[globalRow + globalCol * nCol] = result;\n"
    "}\n";

//verified
void initOccuProb(OccuProb *occuprob){
    double **p = (double**) malloc(sizeof(double*) * occuprob->NbSamples);
    if (!p){
        fprintf(stderr, "Failed to allocate %d number of double* memory blocks.\n", occuprob->NbSamples);
        exit(1);
    } else {
        occuprob->resp = p;
    }
    double *q;
    for (int i = 0; i < occuprob->NbSamples; i++){
        q = (double*) malloc(sizeof(double) * occuprob->NC);
        if (!q){
            fprintf(stderr, "Failed to allocate %d numer of double memory blocks.\n", occuprob->NC);
            exit(1);
        } else {
            occuprob->resp[i] = q;
        }
    }
}
//verified
void clearOccuProb(OccuProb *occuprob){
    for (int i = 0; i < occuprob->NbSamples; i++)
        free(occuprob->resp[i]);
    free(occuprob->resp);
}

//verified
void logsumexp(OccuProb *prob, double *LSE){
    // LSE should be the size of number of samples in the given prob i.e NbSamples
    long double intRes = 0.0f;
    for (int i = 0; i < prob->NbSamples; i++){
        intRes = 0.0f;
        for (int j = 0; j < prob->NC; j++){
            intRes += (long double) expl((long double) prob->resp[i][j]);
        }
        LSE[i] = (double) logl(intRes);
    }
}

//verified
void computeLogWeightProb(MFCCFeatures *mfcc, OccuProb *occuLWprob, double *glbLogDet, double **glbPrec, double **glbMeanPrecProd, double *glbMeanPrecProdSum, double *glbLogWeight){
    long double intRes;
    double normlogPi = ((double) -0.5 * nFeatures * log((2.0f * 3.14159265358979323846f)));
    for (int ns = 0; ns < occuLWprob->NbSamples; ns++){
        for (int nc = 0; nc < occuLWprob->NC; nc++){
            intRes = glbMeanPrecProdSum[nc];
            for (int nf = 0; nf < nFeatures; nf++){
                intRes -= ((long double) 2.0f * mfcc->Features[ns][nf] * glbMeanPrecProd[nf][nc]);
                intRes += ((long double) mfcc->Features[ns][nf] * mfcc->Features[ns][nf] * glbPrec[nc][nf]);
            }
            occuLWprob->resp[ns][nc] = normlogPi - 0.5 * (double) intRes + glbLogDet[nc] + glbLogWeight[nc];
        }
    }
}

//verified
void computeNormWeightProb(OccuProb *occuLWprob){
    //this will calculate the normalized weighted probability of each sample with respect to various components
    int nSamples = occuLWprob->NbSamples;
    double *LSE = (double*) malloc(sizeof(double) * nSamples);
    if (!LSE){
        printf("Could not allocate %d number of double type memory blocks.\n", nSamples);
        exit(1);
    }
    logsumexp(occuLWprob, LSE);
    double norm, x;
    for (int ns = 0; ns < nSamples; ns++){
        norm = LSE[ns];
        for (int nc = 0; nc < occuLWprob->NC; nc++){
            x = occuLWprob->resp[ns][nc] - norm;
            occuLWprob->resp[ns][nc] = exp(x);
        }
    }
    free(LSE);
}

//verified
void computeZeroStat(OccuProb *occuProb, double *zeroStat){
    //zeroStat is the size of number of components
    //can be done by gpu stuffs
    double constant = 2.220446049250313e-15f;
    for (int nc = 0; nc < occuProb->NC; nc++){
        zeroStat[nc] = constant;
        for (int ns = 0; ns < occuProb->NbSamples; ns++){
            zeroStat[nc] += occuProb->resp[ns][nc];
        }
    }
}

//verified
void computeFirstStat(OccuProb *occuProb, double *zeroStat, double **glbSqrtPrec, double **glbMean, double **firstStat, MFCCFeatures *mfcc){
    double intRes;
    // this is the kind of matrix multiplication with specialized norm can be done using gpu stuff
    for (int nc = 0; nc < occuProb->NC; nc++){
        for (int nf = 0; nf < nFeatures; nf++){
            intRes = 0.0f;
            for (int ns = 0; ns < occuProb->NbSamples; ns++){
                intRes += (occuProb->resp[ns][nc] * mfcc->Features[ns][nf]);
            }
            intRes -= (glbMean[nc][nf] * zeroStat[nc]);
            firstStat[nc][nf] = intRes * glbSqrtPrec[nc][nf];
        }
    }
}

//verified
void computePosteriorPrecMat(double *zeroStat, double **precMatrixL, TVMatrix *tvMat){
    //double intRes;
    int NC = tvMat->NC;
    int nCol = tvMat->TVCol;
    //int NF = tvMat->nFeatures;
    //double intRes;
    //int index;

    int nRow = tvMat->TVrow;

    double *MTVGPU = (double*) malloc(sizeof(double) * nRow * nCol);
    if (!MTVGPU){
        printf("Could not allocate %d number of double type memory blocks.\n", nCol * nRow);
        exit(1);
    } else {
        int d;
        for (int i = 0; i < nCol; i++){
            d = i * nRow;
            for (int r = 0; r < nRow; r++){
                MTVGPU[r + d] = tvMat->TV[r][i];
            }
        }
    }
    
    enum Consts {INFOLOG_LEN = 512};
    GLchar infoLog[INFOLOG_LEN];
    //set context for initializing EGL and openGL stuffs here
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returns EGL_NO_DISPLAY.\n");
        exit(1);
    }
    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        exit(1);
    }
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
    //creating compute shader
    GLint success;
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(1);
    }

    glShaderSource(computeShader, 1, &gComputeShaderPosteriorPrecMatrix, NULL);
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
    GLuint matASSbo;
    GLuint matBSSbo;
    GLuint matCSSbo;
    GLuint sizeA, sizeB, sizeC;

    sizeA = (GLuint) nRow * nCol;
    sizeB = (GLuint) NC;
    sizeC = (GLuint) nCol * nCol;

    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeA * sizeof(double), MTVGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeB * sizeof(double), zeroStat, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matCSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeC * sizeof(double), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matCSSbo);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    glUseProgram(shaderProgram);
    glDispatchCompute(5, 5, 1);   // sizeA/local_size_x, sizeB/local_size_y, sizeC/local_size_z
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    double* pOut = (double*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeC * sizeof(double), GL_MAP_READ_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    for (int r = 0; r < nCol; r++){
        for (int c = 0; c < nCol; c++){
            precMatrixL[r][c] = pOut[c * nCol + r];
        }
    }
    /*for (int i = 0; i < nCol; i++){
        printf("%lf\t",precMatrixL[0][i]);
    }
    printf("\n");
    */
    glDeleteShader(computeShader);
    glDeleteBuffers(1, &matCSSbo);
    glDeleteBuffers(1, &matBSSbo);
    glDeleteBuffers(1, &matASSbo);
    glDeleteProgram(shaderProgram);
    eglDestroyContext(dpy, context);
    eglTerminate(dpy);
    /*for (int r = 0; r < 1; r++){
        for (int c = 0; c < nCol; c++){
            printf("%lf\t",precMatrixL[r][c]);
        }
        printf("\n");
    }
    */
    free(MTVGPU);
    //free(pOut);
}

//verified
void computeInversePosteriorPrecMat(double **precMatrixL, double **precInvMatrixL, int len){
    SqMat sqMat;
    sqMat.Row = len;
    initSqMat(&sqMat);
    SqMat invsqMat;
    invsqMat.Row = len;
    initSqMat(&invsqMat);
    for (int i = 0; i < len; i++){
        for (int j = 0; j < len; j++){
            sqMat.matrix[i][j] = precMatrixL[i][j];
        }
    }
    //printf("I am going to compute inverse precision matrix %d.\n",len);
    inverseMatrixLU(&sqMat, &invsqMat);
    //printf("I came after computing inverseMatrix.\n");
    for (int i = 0; i < len; i++){
        for (int j = 0; j < len; j++){
            precInvMatrixL[i][j] = invsqMat.matrix[i][j];
        }
    }
    clearSqMat(&sqMat);
    clearSqMat(&invsqMat);
}

//verified
void computeWVector(double **precInvMatrixL, TVMatrix *tvMat, double **firstStat, double *wVector){
    int nCol = tvMat->TVCol;
    int NF = tvMat->nFeatures;
    int NC = tvMat->NC;
    int nRow = tvMat->TVrow;
    
    //for doing the computation in gpu, we are creating the array to transfer the data into the buffer data objects
    double *precInvMatrixLGPU = (double*) malloc(sizeof(double) * nCol * nCol);
    if (!precInvMatrixLGPU){
        printf("Could not allocate %d number of double type memory blocks.\n",nCol * nCol);
        exit(1);
    } else {
        int d;
        for (int i = 0; i < nCol; i++){
            d = i * nCol;
            for (int r = 0; r < nCol; r++){
                precInvMatrixLGPU[r + d] = precInvMatrixL[r][i];
            }
        }
    }
    double *MatrixTVGPU = (double*) malloc(sizeof(double) * nRow * nCol);
    if (!MatrixTVGPU){\
        printf("Could not allocate %d number of double type memory blocks.\n", nCol * nRow);
        exit(1);
    } else {
        int d;
        for (int c = 0; c < nRow; c++){
            d = c * nCol;
            for (int i = 0; i < nCol; i++){
                MatrixTVGPU[i + d] = tvMat->TV[c][i];
            }
        }
    }

    enum Consts {INFOLOG_LEN = 512};
    GLchar infoLog[INFOLOG_LEN];
    //set context for initializing EGL and openGL stuffs here
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returns EGL_NO_DISPLAY.\n");
        exit(1);
    }
    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        exit(1);
    }
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
    //creating compute shader
    GLint success;
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(1);
    }
    
    glShaderSource(computeShader, 1, &gComputeShaderIntRes, NULL);
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
    GLuint matDSSbo;
    GLuint sizeA, sizeB, sizeC, sizeD;

    sizeA = (GLuint) nCol * nCol;
    sizeB = (GLuint) nRow * nCol;
    sizeC = (GLuint) nCol * nRow;

    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeA * sizeof(double), precInvMatrixLGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeB * sizeof(double), MatrixTVGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matCSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeC * sizeof(double), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matCSSbo);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    glUseProgram(shaderProgram);
    glDispatchCompute(1,19361,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    //glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    double* interMatrixGPU = (double*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeC * sizeof(double), GL_MAP_READ_BIT);
    
    free(precInvMatrixLGPU);
    free(MatrixTVGPU);

    double *flattenFirstStat = (double*) malloc(sizeof(double) * nRow);
    if (!flattenFirstStat){
        printf("Could not allocate %d number of double type memory blocks.\n", nRow);
        exit(1);
    }
    for (int nc = 0; nc < NC; nc++){
        for (int f = 0; f < NF; f++){
            flattenFirstStat[NF * nc + f] = firstStat[nc][f];
        }
    }
    // can be done by gpu processors
    
    glDeleteShader(computeShader);
    glDeleteBuffers(1, &matBSSbo);
    glDeleteBuffers(1, &matASSbo);
    computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(1);
    }

    glShaderSource(computeShader, 1, &gComputeShaderIVector, NULL);
    glCompileShader(computeShader);
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(computeShader, INFOLOG_LEN, NULL, infoLog);
        printf("ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n%s\n", infoLog);
    }
    //creating the shader program
    shaderProgram = glCreateProgram();
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
    sizeA = (GLuint) nCol * nRow;
    sizeB = (GLuint) nRow;
    sizeD = (GLuint) nCol;

    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeA * sizeof(double), interMatrixGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeB * sizeof(double), flattenFirstStat, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matDSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matDSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeD * sizeof(double), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matDSSbo);

    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    glUseProgram(shaderProgram);
    glDispatchCompute(1,1,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matDSSbo);
    double* flattenFirstStatGPU = (double*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeD * sizeof(double), GL_MAP_READ_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    for (int k = 0; k < nCol; k++){
        wVector[k] = flattenFirstStatGPU[k];
        //printf("%lf\t",wVector[k]);
    }
    //printf("\n");
    

    glDeleteShader(computeShader);
    glDeleteBuffers(1, &matCSSbo);
    glDeleteBuffers(1, &matBSSbo);
    glDeleteBuffers(1, &matASSbo);
    glDeleteBuffers(1, &matDSSbo);
    eglDestroyContext(dpy, context);
    eglTerminate(dpy);
    free(flattenFirstStat);

}

//verified
double computeCosineSimilarityScore(double *wTargVector, double *wTestVector, int Len){
    double score = (double) 0.0f;
    double wTargNorm = 2.220446049250313e-15f;
    double wTestNorm = 2.220446049250313e-15f;
    for (int i = 0; i < Len; i++){
        score += wTargVector[i] * wTestVector[i];
        wTargNorm += wTargVector[i] * wTargVector[i];
        wTestNorm += wTestVector[i] * wTestVector[i];
    }
    return score / (wTargNorm * wTestNorm);
}

