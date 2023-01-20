#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include<time.h>


#include <EGL/egl.h>
//#include <EGL/eglext.h>
#include <GLES3/gl31.h>
//#include <GLES3/gl32.h>
//#define GLFW_INCLUDE_ES2



static const char *gComputeShaderMatMul = 
    "#version 320 es\n"
    "layout(local_size_x = 1, local_size_y = 1) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "   double data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "   double data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "   double data[];\n"
    "} matC;\n"
    "shared uint M = 10u;\n"
    "shared uint N = 10u;\n"
    "shared uint K = 10u;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   double acc = double(0.0);\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k*M + globalRow] * matB.data[globalCol * K + k];\n"
    "matC.data[globalRow + globalCol * M] = acc;\n"
    "}\n";

int main(){
    clock_t start, end;
    double execution_time;
    start = clock();
    enum Consts {INFOLOG_LEN = 512};
    GLchar infoLog[INFOLOG_LEN];
    GLint success;
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return 0;
    }
    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        return 0;
    }
    printf("EGL versions %u, %u\n", (unsigned int) majorVersion, (unsigned int) minorVersion);
    
    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_NONE };
    if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
        printf("eglChooseConfig failed\n");
        return 0;
    }

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed\n");
        return 0;
    }
    returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        return 0;
    }
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(0);
    }
    glShaderSource(computeShader, 1, &gComputeShaderMatMul, NULL);
    glCompileShader(computeShader);
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(computeShader, INFOLOG_LEN, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }
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
    
    const GLuint arraySize = 100;
    double *f0 = (double*) malloc(sizeof(double) * arraySize);
    double *f1 = (double*) malloc(sizeof(double) * arraySize);
    //double f0[arraySize];
    //double f1[arraySize]; 
    for (GLuint i = 0; i < arraySize; ++i)
    {
        f0[i] = i;
        f1[i] = 1.0f;
    }
    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(double), f0, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(double), f1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matCSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(double), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, matCSSbo);
        
    /*
    GLuint cN, cM, cK;
    glGenBuffers(1, &cN);
    glBindBuffer(GL_UNIFORM_BUFFER, cN);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(unsigned int), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, cN, 0,sizeof(unsigned int));
    glGenBuffers(1, &cM);
    glBindBuffer(GL_UNIFORM_BUFFER, cM);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(unsigned int), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, cM, 0,sizeof(unsigned int));
    glGenBuffers(1, &cK);
    glBindBuffer(GL_UNIFORM_BUFFER, cK);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(unsigned int), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, cK, 0,sizeof(unsigned int));
    */
    GLenum err = glGetError(); 
    if (err != GL_NO_ERROR) 
        printf("glGetError returns %d\n", err); 
    start = clock();
    glUseProgram(shaderProgram);
    glDispatchCompute(100,100,1);   // arraySize/local_size_x
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    double* pOut = (double*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, arraySize * sizeof(double), GL_MAP_READ_BIT);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation\n",execution_time);
    //free(pOut);
    double *out = (double*) malloc(sizeof(double) * arraySize);
    /*
    start = clock();
    for (int i = 0; i < arraySize / 1000; i++){
        for (int k = 0; k < arraySize / 1000; k++){
            for (int j = 0; j < arraySize / 1000; j++){
                out[1000 * j + i] += f0[1000 * k + i] * f1[1000 * j + k];
            }
        }
    }
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation\n",execution_time);
    */
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            printf("%lf\t",pOut[j + i * 10]);
        }
        printf("\n");
    }
    eglDestroyContext(dpy, context);
    eglTerminate(dpy);
    
    GLuint matShader = glCreateShader(GL_COMPUTE_SHADER);
    
    return 0;
}
