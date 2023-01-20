#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include<time.h>


#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#define GLFW_INCLUDE_ES2



static const char *gComputeShaderMatMul = 
    "#version 320 es\n"
    "layout(local_size_x = 10, local_size_y = 10) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "   float data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "   float data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "   float data[];\n"
    "} matC;\n"
    "shared uint M = 1000u;\n"
    "shared uint N = 1000u;\n"
    "shared uint K = 1000u;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   float acc = 0.0;\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k + K * globalRow] * matB.data[globalCol +  K * k];\n"
    "matC.data[globalRow * K + globalCol] = acc;\n"
    "}\n";

int main(){
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
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
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
    
    const GLuint arraySize = 1000000;
    float f0[arraySize];
    float f1[arraySize]; 
    for (GLuint i = 0; i < arraySize; ++i)
    {
        f0[i] = i;
        f1[i] = i;
    }
    clock_t start, end;
    double execution_time;
    start = clock();
    glGenBuffers(1, &matASSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matASSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), f0, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matASSbo);

    glGenBuffers(1, &matBSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), f1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matBSSbo);

    glGenBuffers(1, &matCSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matCSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), NULL, GL_STATIC_DRAW);
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
    float* pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, arraySize * sizeof(float), GL_MAP_READ_BIT);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation\n",execution_time);
    //free(pOut);
    //float *out = (float*) malloc(sizeof(float) * arraySize);
    start = clock();
    /*
    for (int i = 0; i < arraySize / 1000; i++){
        for (int k = 0; k < arraySize / 1000; k++){
            for (int j = 0; j < arraySize / 1000; j++){
                out[1000 * j + i] += f0[1000 * k + i] * f1[1000 * j + k];
            }
        }
    }*/
    for (int i = 0; i < arraySize / 1000; i++)
        printf("%lf\t",pOut[i]);
    printf("\n\n");
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation\n",execution_time);
    eglDestroyContext(dpy, context);
    eglTerminate(dpy);
    
    GLuint matShader = glCreateShader(GL_COMPUTE_SHADER);
    
    return 0;
}
