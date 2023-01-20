#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include<time.h>
//#include <glad/glad.h>

//#include <EGL/egl.h>

//#include <EGL/eglext.h>
//#include <GLES2/gl2.h>
//#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#include <GLFW/glfw3.h>
#define GLFW_INCLUDE_ES2
#include <stdbool.h>

static const GLuint WIDTH = 800;
static const GLuint HEIGHT = 600;

const unsigned int SCREEN_WIDTH = 600;
const unsigned int SCREEN_HEIGHT = 400;
bool vSync = true;
static const char *gComputeShader =
    "#version 320 es\n"
    "layout(local_size_x = 1000) in;\n"
    "layout(std430, binding = 0) readonly buffer Input0 {\n"
    "    float data[];\n"
    "} input0;\n"
    "layout(std430, binding = 1) readonly buffer Input1 {\n"
    "    float data[];\n"
    "} input1;\n"
    "layout(std430, binding = 2) writeonly buffer Output {\n"
    "    float data[];\n"
    "} output0;\n"
    "void main()\n"
    "{\n"
    "    uint idx = gl_GlobalInvocationID.x;\n"
    "    float f = input0.data[idx] * input1.data[idx];"
    "    output0.data[idx] = f;\n"
    "}\n";


static const char *gComputeShaderMatMul = 
    "#version 320 es\n"
    "layout(local_size_x = 8, local_size_y = 8) in;\n"
    "layout(std430, binding = 0) readonly buffer MatA {\n"
    "   float data[];\n"
    "} matA;\n"
    "layout(std430, binding = 1) readonly buffer MatB {\n"
    "   float data[];\n"
    "} matB;\n"
    "layout(std430, binding = 2) writeonly buffer MatC {\n"
    "   float data[];\n"
    "} matC;\n"
    "uniform uint M;\n"
    "uniform uint N;\n"
    "uniform uint K;\n"
    "void main(){\n"
    "   uint globalRow = gl_GlobalInvocationID.x;\n"
    "   uint globalCol = gl_GlobalInvocationID.y;\n"
    "   float acc = 0.0;\n"
    "   for (uint k = 0u; k < K; k++)\n"
    "       acc += matA.data[k*M + globalRow] * matB.data[globalCol * K + k];\n"
    "matC.data[globalRow + globalCol * M] = acc;\n"
    "}\n";

int main(){
    enum Consts {INFOLOG_LEN = 512};
    GLchar infoLog[INFOLOG_LEN];
    GLint success;
    GLuint shader_program, vbo;
    GLint pos;
    GLFWwindow* window;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);//2
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(WIDTH, HEIGHT, __FILE__, NULL, NULL);
    glfwMakeContextCurrent(window);

    printf("GL_VERSION  : %s\n", glGetString(GL_VERSION) );
    printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER) );

    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (computeShader == 0){
        printf("Could not create compute shader.\n");
        exit(0);
    }
    glShaderSource(computeShader, 1, &gComputeShader, NULL);
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
    GLuint input0SSbo;
    GLuint input1SSbo;
    GLuint outputSSbo;
    
    const GLuint arraySize = 80000; // 800
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
    glGenBuffers(1, &input0SSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input0SSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), f0, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input0SSbo);

    glGenBuffers(1, &input1SSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input1SSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), f1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, input1SSbo);

    glGenBuffers(1, &outputSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, outputSSbo);

    GLenum err = glGetError(); 
    if (err != GL_NO_ERROR) 
        printf("glGetError returns %d\n", err); 
    start = clock();
    glUseProgram(shaderProgram);
    glDispatchCompute((int)arraySize/1000,1,1);   // arraySize/local_size_x
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    err = glGetError();
    if (err != GL_NO_ERROR)
        printf("glGetError returns %d\n", err);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSbo);
    float* pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, arraySize * sizeof(float), GL_MAP_READ_BIT);
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation in gpu.\n",execution_time);
    start = clock();
    for (int i = 0; i < arraySize; i++)
        pOut[i] = f0[i] * f1[i];
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("It tooks around %lf time to do the computation in cpu\n",execution_time);
    //eglDestroyContext(dpy, context);
    //eglTerminate(dpy);
    
    GLuint matShader = glCreateShader(GL_COMPUTE_SHADER);

    glShaderSource(matShader, 1, &gComputeShaderMatMul, NULL);
    glCompileShader(matShader);
    glGetShaderiv(matShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(matShader, INFOLOG_LEN, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }


    return 0;
}
