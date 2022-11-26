import numpy
from .melfc_cepstra import *
from .lpc_cepstra import *

"""This code is reviewed once
    and tested few times
"""
__license__ = "KIWIP Tech SAS"
__status__ = "In Production"
__email__ = "abdul.n@kiwip.TECH"
__maintainer__ = "Abdul Majith Noordheen"

def power_spectrum(audio,samplerate=16000,winlen=0.025,winstep=0.01,preemph=0.97,nfft=None,lowfreq=0,highfreq=None,nfilt=26):
    """For the given audio signal and the respective parameter, following code computes the energy spectrum for framed input audio
    Methodology: do pre emphasis (low pass filter) on the input audio signal
    frame (split) the given audio signal into overlapping multiple signal
    do the windowing on each frame to nullify the boundary effects
    do the discrete fourier transformation on each frame.

    Compute the filterbanks for the range of low to high frequency in the linear scale of mel spectrum
    compute the energy spectrum of given audio signal with respect to the filter bank
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    highfreq = highfreq or samplerate / 2
    emp_signal = preemphasis(audio,preemph)
    framed_sig = get_frames(emp_signal, samplerate, winlen, winstep)
    wframed_sig = windowing_frames(framed_sig)
    frq_feat = apply_dft_spectrum(wframed_sig, nfft)
    frq_feat = numpy.square(numpy.absolute(frq_feat)) / nfft
    """ compute the filterbanks for the given number of filters
    """
    fbanks = filterbanks(nfilt=nfilt, nfft=nfft, samplerate=samplerate, lowfreq=lowfreq, highfreq = highfreq)
    #compute energy banks
    eng_spect = energy_spectrum(frq_feat, fbanks)
    return eng_spect

def apply_rastafilter(signal):
    """Given framed signal, following code do the rasta filter on the framed input signal
    """
    signal[numpy.where(signal == 0)] = numpy.finfo(float).eps
    sig = numpy.log(signal)
    sig = rastafilt(sig)
    return numpy.exp(sig)

def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies
    """
    return 6. * numpy.arcsinh(f / 600.)

def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)
    """
    return 600. * numpy.sinh(z / 6.)

def find_lpcept(signal,p=12,Q=None):
    """Given the energy spectrum, following code compute the cepstrum of framed energy spectrum 
    methdology: compute inverse fourier transform for each frame of energy spectrum
    compute the linear predictive coding coefficeints on the band of inverse fourier energy spectrum 
    compute the cepstrum from the LPC co-efficeints
    """
    Q = Q or int(p * 3 / 2)
    Nf, nbands = signal.shape[0], signal.shape[1]
    isignal = numpy.real(numpy.fft.ifft(numpy.hstack((signal,signal[:,numpy.arange(nbands-2,0,-1)]))))
    isignal = isignal[:, :nbands]
    Y = numpy.zeros((Nf, Q))
    for f in range(Nf):
        cor_f = auto_correlation(isignal[f,:], p)
        _, f_lpc, _ = lpc_analysis(cor_f, p)
        f_lpcc = lpc_cepstral(f_lpc, Q)
        Y[f,:] = f_lpcc[1:]
    return Y

def get_PLP(audioSignal,samplerate=16000,winlen=0.025,hoplen=0.01,emph_coeff=0.95,nfft=None,lowfreq=0,highfreq=None,nfilt=None,NORM=None,p=12,Q=18):
    """ Given the audio signal, compute the Perpetual linear prediction
    Methodology first compute the energy spectrum of input audio signal (framed)
    compute the rasta filter based on the option
    finally compute the lpc of the framed inverse fourier energy spectrum
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    highfreq = highfreq or samplerate / 2
    nfilt = nfilt or int(numpy.ceil(hz2bark(samplerate / 2)) + 1)
    eng_spect = power_spectrum(audioSignal,samplerate = samplerate,winlen = winlen,winstep = hoplen,preemph = emph_coeff,nfft = nfft,lowfreq = lowfreq,highfreq = highfreq,nfilt=nfilt)
    if NORM == 'RASTA':
        eng_spect = apply_rastafilter(eng_spect)
    cepstrum = find_lpcept(eng_spect, p=p, Q = Q)
    cepstrum = parameter_weighting(cepstrum)
    return cepstrum

def PLPextraction(audio,sr,winlen=0.25,hoplen=0.01,emph_coeff=0.95,nfft=None,lowfreq=0,highfreq=None,nfilt=None,NORM =None,p=12,Q=18,Delta='Yes',K=3):
    plp = get_PLP(audio,sr,winlen,hoplen,emph_coeff,nfft,lowfreq,highfreq,nfilt,NORM,p,Q)
    if Delta == 'Yes':
        delta_plp = get_delta_LPCC(plp,K)
        plp = numpy.hstack((plp,delta_plp))
    if NORM == 'CMS':
        plp = CMNorm(plp)
    else: 
        if NORM == 'CMVN':
            plp = CMVar_norm(plp)
    return plp
