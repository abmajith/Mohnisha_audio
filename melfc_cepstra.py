import numpy
import warnings
from .lpc_cepstra import *
warnings.filterwarnings("ignore")

"""This code is reviewed once and runed the test code once
    except rasta filter
"""

__license__ = "KIWIP Tech SAS"
__status__ = "Production"
__email__ = "abdul.n@kiwip.TECH"
__maintainer__ = "Abdul Majith"

def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.

    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = int(winlen * samplerate)
    nfft = 1
    for i in range(0,window_length_samples):
        nfft *= 2
        if nfft >= window_length_samples:
            break
    return nfft

def filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels scale
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)

    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bins = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
    return fbank

def apply_dft_spectrum(framed_signal, NFFT):
    if framed_signal.shape[1] > NFFT:
        logging.warn('frame length (%d) is greater than FFT size (%d),\n frame will be truncated \n \t (leads to loss of information (precisition)).\n Increase NFFT to avoid truncations.\n',framed_signal.shape[1], NFFT)
    complex_spec = numpy.fft.rfft(framed_signal, NFFT)
    return complex_spec

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def energy_spectrum(pow_spect, fbanks):
    nf = pow_spect.shape[0]
    if nf <= 0:
        raise ValueError("No data to compute energy spectrum\n")
    feat = numpy.dot(pow_spect,fbanks.T)
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat)
    return feat

def get_dct2(feng, ncept = 19):
    Nf, ne = feng.shape[0], feng.shape[1]
    cosine = numpy.zeros((ncept, ne))
    cosine[0,:] = numpy.cos( (numpy.pi * 0 / (2 * ne)) *  ((2 * numpy.array(range(ne))) + 1) ) / numpy.sqrt(ne)
    for i in range(1, ncept):
        cosine[i,:] =  numpy.cos( (numpy.pi * i / (2 * ne)) *  ((2 * numpy.array(range(ne))) + 1) ) / numpy.sqrt(ne / 2)
    feat = numpy.dot(feng, cosine.T)
    return feat

def lifter(cepstra, L=22):
    """Apply a cepstral lifter to the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.) * numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def get_MFCC(audio,samplerate=16000,winlen=0.02,winstep=0.01,numcep=19,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.96,ceplifter=22,NORM=None):
    nfft = nfft or calculate_nfft(samplerate, winlen)
    """ Here we are computing the Mel Frequency Cepstral Co-efficients
    for the given audio signal of one dimenstional array of size N
    wit sample rate default 16000, 
    extracting the mfcc with frame interval default 0.025 milli seconds, 
    frame difference of default 0.01 milli seconds, with default 13 number of cepstrum
    parameter lowfreq defines the min frequency, and highfreq max frequency to be considered in extracting the 
    cepstrum co-efficients

    """
    highfreq= highfreq or samplerate/2
    """Methedology original sound ==> preemphasis
    get frames of preemphasized signal
    frames of signal ==> window the frames to nullify the boundary effects
    windowed frames ==> discrete fast fourier transformation and compute the normalized power in each frequency region
    """
    emp_signal = preemphasis(audio,preemph)
    framed_sig = get_frames(emp_signal, samplerate, winlen, winstep)
    wframed_sig = windowing_frames(framed_sig)
    frq_feat = apply_dft_spectrum(wframed_sig, nfft)
    frq_feat = numpy.square(numpy.absolute(frq_feat)) / nfft
    """ compute the filterbanks for the given number of filters

    """
    fbanks = filterbanks(nfilt=nfilt, nfft=nfft, samplerate=samplerate, lowfreq=lowfreq, highfreq = highfreq)
    #compute the energy banks 
    eng_spect = energy_spectrum(frq_feat, fbanks)

    log_eng = numpy.log(eng_spect)
    if NORM == 'RASTA':
        log_eng = rastafilt(log_eng)
    #compute the discrete cosine transformation
    MFCC = get_dct2(log_eng, numcep)
    MFCC = lifter(MFCC, ceplifter)
    MFCC[numpy.where(MFCC == 0)] = numpy.finfo(float).eps
    return MFCC

def rastafilt(signal):
    numer = numpy.array([0.2,0.1,0.,-0.1,-0.2])
    denom = numpy.array([1,-0.98])
    if signal.shape[0] <= 4:
        logging.warn('Input number of frames (%d) is less than 5,\n all the frames will be zero after applying rasta Filter\n', signal.shape[0])
        return numpy.zeros((4,signal.shape[1]))
    Z = numpy.zeros((4,signal.shape[1]))
    Z[0,:] = 0.2 * signal[0,:]
    Z[1,:] = 0.2 * signal[1,:] + 0.1 * signal[0,:]                      + 0.98 * Z[0,:]
    Z[2,:] = 0.2 * signal[2,:] + 0.1 * signal[1,:]                      + 0.98 * Z[1,:]
    Z[3,:] = 0.2 * signal[3,:] + 0.1 * signal[2,:] - 0.1  * signal[0,:] + 0.98 * Z[2,:]

    out = numpy.zeros((signal.shape[0], signal.shape[1]))
    out[0:4,:] = Z
    for i in range(4,signal.shape[0]):
        out[i,:] = 0.98 * out[i-1,:] + 0.2 * signal[i,:] + 0.1 * signal[i-1,:] - 0.1 * signal[i-3,:] - 0.2 * signal[i-4,:]
    out[0:4,:] = numpy.zeros((4,signal.shape[1]))
    return out

def MFCCextraction(audio,sr,winlen=0.02,winstep=0.01,numcep=19,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.96,ceplifter=22,NORM=None,Delta = 'No',K=3):
    mfcc = get_MFCC(audio,sr,winlen,winstep,numcep,nfilt,nfft,lowfreq,highfreq,preemph,ceplifter,NORM=NORM)
    if Delta == 'Yes':
        delta_mfcc = get_delta_LPCC(mfcc,K)
        mfcc = numpy.hstack((mfcc,delta_mfcc))
    if NORM == 'CMS':
        mfcc = CMNorm(mfcc)
    else: 
        if NORM == 'CMVN':
            mfcc = CMVar_norm(mfcc)
    return mfcc



