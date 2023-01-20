#include <stdio.h>
#include <math.h>
#include <complex.h>


double PI;
typedef double complex cplx;

void _fft(cplx buf[], cplx out[], int n, int step){
    if (step < n){
        _fft(out, buf, n, step * 2);
        _fft(out + step, buf + step, n, step * 2);
        for (int i = 0; i < n; i += 2 * step){
            cplx t = cexp(-I * PI * i / n) * out[i + step];
            buf[i / 2] = out[i] + t;
            buf[ (i + n) / 2] = out[i] - t;
        }
    }
}

void _rfft(double rbuf[], double ibuf[], double rout[], double iout[], int n, int step){
    if (step < n){
        _rfft(rout, iout, rbuf, ibuf, n, step * 2);
        _rfft(rout + step, iout + step, rbuf + step, ibuf + step, n, step * 2);
        double rt, it, c, s;
        double piconst = 3.141592653589793f;
        for (int i = 0; i < n; i += 2 * step){
            c = cos(piconst * ((double) i / n));
            s = sin(piconst * ((double) i / n));
            rt = c * rout[i + step] + s * iout[i + step];
            it = iout[i + step] * c - rout[i + step] * s;
            rbuf[i / 2] = rout[i] + rt;
            ibuf[i / 2] = iout[i] + it;
            rbuf[(i + n) / 2] = rout[i] - rt;
            ibuf[(i + n) / 2] = iout[i] - it;
        }
    }
}


void fft(cplx buf[], int n){
    cplx out[n];
    for (int i = 0; i < n; i++) out[i] = buf[i];
    _fft(buf, out, n, ((int) 1));
}


void rfft(double rbuf[], double ibuf[], int n){
    double rout[n];
    double iout[n];
    for (int i = 0; i < n; i++){
        rout[i] = rbuf[i];
        iout[i] = ibuf[i];
    }
    _rfft(rbuf, ibuf, rout, iout, n, 1);
}

void show(const char *s, cplx buf[], int len){
    printf("%s",s);
    for (int i = 0; i < len; i++){
        if (!cimag(buf[i]))
            printf("(%g + j %lf) ", creal(buf[i]), (double) 0.0f);
        else
            printf("(%g + j %g) ", creal(buf[i]), cimag(buf[i]));
    }
    printf("\n");
}


void rshow(const char *s, double rbuf[], double ibuf[], int len){
    printf("%s",s);
    for (int i = 0; i < len; i++){
        printf("(%lf + %lf i) ", rbuf[i], ibuf[i]);
    }
    printf("\n");
}

int main()
{
	PI = atan2(1, 1) * 4;
	cplx buf[] = {1, 1, 1, 1, 0, 0, 0, 0};

	show("Data: ", buf, 8);
	fft(buf, 8);
	show("\nFFT : ", buf, 8);

    double rbuf[] = {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    double ibuf[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    rshow("rData: ", rbuf, ibuf, 8);
    rfft(rbuf, ibuf, 8);
    rshow("rFFT: ", rbuf, ibuf, 8);

	return 0;
}
