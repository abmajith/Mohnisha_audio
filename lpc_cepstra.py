import numpy
"""This code is reviewed once
    and tested few times
"""
__license__ = "KIWIP Tech SAS"
__status__ = "In Production"
__email__ = "abdul.n@kiwip.TECH"
__maintainer__ = "Abdul Majith Noordheen"

def preemphasis(signal, coeff = 0.95):
    """ Perfrom preemphasis (a FIR filter) on the input signal
    signal is a N by 1 (1d array)
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

def get_frames(audioSignal, sampleRate= 16000, winLen = 0.02, winStep = 0.01):
    """ given audio Signal of dim N by 1 (1d array), sampleRate of the given audio signal
    for the given paramter of frame length winLen in mili seconds, and 
    frame interval winStep in mili seconds, 
    returns the framed signal a 2d array, each row is a single frame datas 
    """
    speech_length = len(audioSignal)
    frame_length = int(numpy.fix(winLen * sampleRate))
    if frame_length > speech_length:
        raise ValueError("Short utternance, audioSignal is shorter than frame length\n")
    next_frame_index = int(numpy.fix(winStep * sampleRate))
    number_frames = int(numpy.ceil(speech_length - frame_length + next_frame_index) / next_frame_index)
    if speech_length < next_frame_index:
        raise ValueError("Short utternance, audioSignal is shorter than frame distance\n")
    req_speech_length = (number_frames - 1) * next_frame_index + frame_length

    if speech_length < req_speech_length:
        signal = numpy.concatenate((audioSignal, numpy.zeros(req_speech_length - speech_length)))
    else:
        signal = audioSignal

    index = numpy.tile(numpy.arange(0,frame_length), (number_frames,1))  + numpy.tile(numpy.arange(0,(number_frames)*next_frame_index, next_frame_index), (frame_length,1)).T
    index = numpy.array(index, dtype = numpy.int64)
    return signal[index]

def windowing_frames(framed_sig):
    """Given the frames (a 2d matrix, each row is a single frame) of audio signal
    applying windows to reduce the boundary effects
    """
    N = framed_sig.shape[1] # having the number of data points in a single frame
    vect = 2 * numpy.pi * numpy.arange(0,N) / (N-1)
    window = 0.54 - 0.46 * numpy.cos(vect)
    return framed_sig * window

def auto_correlation(frame_data, p = 12):
    """Given data for a single frame compute the auto correlation
    """
    N = len(frame_data)
    if N <= 0 or N < p:
        raise ValueError("There is no (or enough) data in the given frame of function auto_correlation\n")
    R = numpy.zeros((p+1,))
    R[0] = numpy.sum( numpy.square(frame_data) )
    for i in range(1,p+1):
        R[i] = numpy.sum( frame_data[:-i] * frame_data[i:])
    return R

def lpc_analysis(R, p = 12):
    #R = R[0],R[1],....R[p]
    """Given auto correlation values, compute the linear predict coding co-efficients using Durbin's method
    """
    if len(R) != p + 1:
        raise ValueError("There is mismatch in lpc_analysis functions (number of datas and p)\n")
    E = numpy.zeros((p+1,))
    K = numpy.zeros((p,))
    A = numpy.zeros((p,p))
    E[0] = R[0]
    # I did the initialization
    
    #set the first iteration values directly
    K[0] = R[1] / E[0]
    A[0,0] = K[0]
    E[1] = (1 - numpy.square(K[0])) * E[0]

    for i in range(2,p+1):
        J = numpy.array((range(1,i)))
        cf = numpy.sum( A[i-2,J-1] * R[i-J] )
        K[i-1] = (R[i] - cf) / E[i-1] 
        #-1 in the K becuase K indices starts from 1, -1 in E becasue K th value depedns on previous value of E
        A[i-1, i-1] = K[i-1]
        A[i-1,J-1] = A[i-2,J-1] - K[i-1] * A[i-2,i-J-1]
        E[i] = (1 - numpy.square(K[i-1]))  * E[i-1]
    return E, A[p-1,:], K

def lpc_cepstral(coef, cep_nu = 18):
    """Given lpc co-efficients, compute the cepstrum
    """
    cepstrum = numpy.zeros((cep_nu + 1,))
    p = len(coef)
    cepstrum[0] = numpy.log(p)
    if cep_nu <= 0:
        raise ValueError("can't compute lpc cepstrum, number of co-efficients can be less than zero")
    if cep_nu > p:
        cepstrum[1] = coef[0]
        for m in range(2,p+1):
            M = numpy.array(range(1,m))
            cepstrum[m] = coef[m-1] + numpy.sum( cepstrum[M] * coef[m-M-1] * M / m )
        for m in range(p+1, cep_nu+1):
            M = numpy.array(range(m-p,m))
            cepstrum[m] = numpy.sum( cepstrum[M] * numpy.flip(coef) * M / m)
    else:
        cepstrum[1] = coef[0]
        for m in range(2,cep_nu+1):
            M = numpy.array(range(1,m))
            cepstrum[m] = coef[m-1] + numpy.sum(cepstrum[M] * coef[m-M-1] * M / m )
    return cepstrum

def parameter_weighting(frame_ceps):
    """Given the frames of cepstrum, apply the parameter weighting  to achieve robustness
    """
    q = frame_ceps.shape[1]
    window = 1 + ( (q / 2) * numpy.sin(numpy.pi * numpy.arange(1,q+1)/ q))
    return frame_ceps * window

def get_LPCC(audioSignal, samplerate = 16000, winlen = 0.025, hoplen = 0.01, Q = 18, p = 12, emph_coeff = 0.95):
    """For the given audioSignal with sample rate, 
    extract the lpc cepstrum based on the book "Fundamentals of speech recognition" by 
    Lawrence Rabiner and Biing-Hwang Juang
    with given parameter of frame length (winlen), frame distance (hoplen) in milliseconds
    Q represents number of cepstrum needed, p represents number of lpc co-efficients
    """
    N = len(audioSignal)
    if N == 0:
        raise ValueError("there is no data in get_LPCC function\n")
    if (winlen * samplerate > N) or (hoplen * samplerate > N):
        raise ValueError("Too short utternance to get LPCC\n")
    """Methodology
    audiosignal ==> preemphasis the audio
    preemphasised audio ==> frame blocking creating the number of frames for a single utternance 
    framed audio signal ==>  windowing (to nullify the signal discontinuties)
    windowed framed signal ==> autocorrelation
    autocorrelated signal ==> lpc analysis (co-efficients finding)
    lpc co-efficients ==> cepstral co-efficients
    cepstral co-ef ==> parameter weighting (to nullify the noise in the higher order cepstral coefficients also reduce the sensitivity of low-order cepstral coeff)
    """
    # consider the emph_coeff in the range of [0.9, 1]
    emph_audio = preemphasis(audioSignal, emph_coeff)
    # consider winlen between [0.02, 0.45] and hoplen between [0.005, 0.015] but don't have to be strict
    framed_signal = get_frames(emph_audio, samplerate, winlen, hoplen)
    #windowing
    fw_signal = windowing_frames(framed_signal)

    #autocorrelation, consider the p to be less than Q, useually Q = (3 / 2) * p
    Nf = fw_signal.shape[0]
    Fr_lpcc = numpy.zeros((Nf, Q))
    for f in range(Nf):
        cor_f = auto_correlation(fw_signal[f,:], p)
        _, f_lpc, _ = lpc_analysis(cor_f, p)
        f_lpcc = lpc_cepstral(f_lpc, Q)
        Fr_lpcc[f,:] = f_lpcc[1:]
    #Parameter Weighting
    F_LPCC = parameter_weighting(Fr_lpcc)
    return F_LPCC


def get_delta_LPCC(fr_lpcc, K = 3):
    """Given the frame of lpcc, find the deriviaties of cepstrum with approximation parameter K
    """
    N = fr_lpcc.shape[0] # number of frames
    Q = fr_lpcc.shape[1] # number of cepstrum in each frames
    if N <= 0:
        raise ValueError("There is no data in the get_delta_LPCC function\n")
    if K <= 1:
        raise ValueError("K must be an interger >= 1 in get_delta_LPCC function\n")
    K = int(K)
    S = 3 / ( K * (K + 1) * (2 * K + 1) ) # normalization factor
    

    Lpc = numpy.zeros(( 2 * K + N , Q)) # padding zeros top and bottom to have computation later 
    for i in range(1,K+1,1): # to eliminate the sudden spike in the deltas at the boundaries
        Lpc[i-1,:] = fr_lpcc[0,:]
        Lpc[-i,:] = fr_lpcc[-1,:]

    Lpc[K:-K,:] = fr_lpcc
    dlp = numpy.zeros((N, Q))
    for fp in range(K,N+K,1):
        s = numpy.zeros((Q,))
        for k in range(-K,0,1):
            s += k * Lpc[fp + k,:]
            s -= k * Lpc[fp - k,:]
        dlp[fp - K,:] = s
    return dlp * S


#some straigt forward normalization functions are here (when we dont have large data sets to normalize)
def CMNorm(fr_lpcc):
    """Given the rows of cepstrum vector, compute the mean cepstrum vector and substract from each cepstrum
    """
    mu_ = numpy.mean(fr_lpcc, axis = 0, dtype=numpy.float64)
    return fr_lpcc - mu_

def CMVar_norm(fr_lpcc):
    """Given the rows of cepstrum vector. compute the means and variance cepstrum vector and normalize with respect to variance
    result will be zero mean vector of cepstrum with variance of 1
    """
    mu_, nu_ = numpy.mean(fr_lpcc, axis = 0, dtype=numpy.float64), numpy.std(fr_lpcc, axis = 0, dtype=numpy.float64)
    return (fr_lpcc - mu_ ) / nu_

def LPCCextraction(audio,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM=None,Delta='Yes',K=3):
    lpcc = get_LPCC(audio,sr,winlen,hoplen,Q,p,emph_coeff)
    if Delta == 'Yes':
        delta_lpcc = get_delta_LPCC(lpcc, K)
        lpcc = numpy.hstack((lpcc,delta_lpcc))
    if NORM == 'CMS':
        lpcc = CMNorm(lpcc)
    else: 
        if NORM == 'CMVN':
            lpcc = CMVar_norm(lpcc)
    return lpcc

#------------------------------------------------------------------------
#Energy based end clipping of audio signals
def end_voice_clipping_by_Eng(audio,factor=3):
    std = factor * 10 * numpy.var(audio)
    indices = numpy.where(audio > std)[0]
    return indices[0], indices[-1] #start indices and end indices

def end_voice_clipping_by_Thr(audio, thr=100):
    indices = numpy.where(audio > thr)[0]
    return indices[0], indices[-1]
#one can also clip the features if there is a significant gap between the phrases or words, but allowing little bit of 
# silence might help the modelling stuff (I think it will increase the robustness)
#------------------------------------------------------------------------
