import numpy
import warnings
warnings.filterwarnings("ignore")
import os
import tables
import errno
import sys
sys.path.append('../')
from pydub import AudioSegment
from scipy.io.wavfile import read
import time
from sklearn.cluster import KMeans
import random
from sklearn.mixture import GaussianMixture
from multiprocessing import Process, Queue, shared_memory, Pool
from scipy.special import logsumexp

SEG_SIZE = 100000


#-------------------------------------------------------------------------------
#front_end_mfcc extractions
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
def parameter_weighting(frame_ceps):
    """Given the frames of cepstrum, apply the parameter weighting  to achieve robustness
    """
    q = frame_ceps.shape[1]
    window = 1 + ( (q / 2) * numpy.sin(numpy.pi * numpy.arange(1,q+1)/ q))
    return frame_ceps * window




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



#collection features
def compute_STATS_feature(file_path = 'some_h5_features_file'):
    if os.path.isfile(file_path) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    f = tables.open_file(file_path,mode = 'r')
    N_f = f.root.Features.shape[0]
    N_seg = int(numpy.floor(N_f / SEG_SIZE))
    L_end_seg = N_f - SEG_SIZE * N_seg
    c_mean = numpy.zeros((N_seg, f.root.Features.shape[1]))
    for i in range(N_seg):
        c_mean[i,:] = numpy.mean(f.root.Features[i*SEG_SIZE:(i+1)*SEG_SIZE,:],axis = 0, dtype=numpy.float64)
    e_mean = numpy.mean(f.root.Features[N_seg*SEG_SIZE:,:],axis = 0, dtype=numpy.float64)
    scale = SEG_SIZE / N_f
    e_scale = L_end_seg / N_f
    f_mean = ( scale * numpy.sum(c_mean, axis=0, dtype=numpy.float64) ) + ( e_scale * e_mean )

    c_var = numpy.zeros((N_seg, f.root.Features.shape[1]))
    for j in range(N_seg):
        c_var[j,:] = numpy.mean(numpy.square(f.root.Features[j*SEG_SIZE:(j+1)*SEG_SIZE,:]), axis=0, dtype=numpy.float64)
    e_var = numpy.mean(numpy.square(f.root.Features[N_seg*SEG_SIZE:,:]), axis=0, dtype=numpy.float64)
    f_var = ( scale * numpy.sum(c_var, axis=0, dtype=numpy.float64) ) + ( e_scale * e_var )
    f.close()
    return f_mean, numpy.sqrt(f_var,dtype='float64')

#Use the above stats to produce the pre normalized features set from the features of voices
def normalize_features_file(f_mean, source_file = 'some_features_h5_file', dest_file = 'some_norm_h5_file', NORM = 'CMNV', f_std = numpy.array([1,2])):
    # this NORM should in either CMNV or CMS type, if it is CMNV then f_std parameter should be given
    print(f"{source_file}")
    if os.path.isfile(source_file) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source_file)
    if (NORM != 'CMS') and (NORM != 'CMNV'):
        raise ValueError(f"Given Norm options {NORM} is not recognizable, it should be either CMS or CMNV,\n")
    #open the source file in read mode
    s_f = tables.open_file(source_file, mode = 'r')
    N_f = s_f.root.Features.shape[0]
    dim = s_f.root.Features.shape[1]
    #open the destination file to write normalized features
    d_f = tables.open_file(dest_file, mode = 'w')
    array_vect = d_f.create_earray(d_f.root,'Norm_features',tables.Float64Atom(), (0,dim) )
    N_seg = int(numpy.floor(N_f / SEG_SIZE))
    if (NORM == 'CMS'):
        if f_mean.shape[0] != dim:
            raise ValueError(f"Given mean  dimenstion {f_mean.shape[0]} not matching with features dimention {dim}.\n")
        for i in range(N_seg):
            i_f = s_f.root.Features[i*SEG_SIZE:(i + 1)*SEG_SIZE,:]
            array_vect.append(i_f - f_mean)
        i_f = s_f.root.Features[N_seg*SEG_SIZE:,:]
        array_vect.append(i_f - f_mean)
    else:
        if f_std.shape[0] != dim:
            raise ValueError(f"Given standard deviation dimenstion {f_std.shape[0]} not matching with features dimention {dim}.\n")
        for i in range(N_seg):
            i_f = s_f.root.Features[i*SEG_SIZE:(i + 1)*SEG_SIZE,:]
            array_vect.append( (i_f - f_mean) / f_std )
        i_f = s_f.root.Features[N_seg*SEG_SIZE:,:]
        array_vect.append( (i_f - f_mean) / f_std )
    s_f.close()
    d_f.close()
    print(f"Succesfully normalized with option {NORM} to the given features file and stored in {dest_file}.\n")
    return True

def global_mfcc_features_file(folder_name="test_folder",dest_file="some_h_file",dim=38,sampling_rate=32000):
    if os.path.isdir(folder_name) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), folder_name)
    f = tables.open_file(dest_file, mode = 'w')
    array_vect = f.create_earray(f.root,'Features',tables.Float64Atom(), (0,dim) )
    ls_files = os.listdir(folder_name)
    I = 0
    for filename in ls_files:
        name_file = os.path.join(folder_name, filename)
        sound = AudioSegment.from_file(name_file,format="mp3")
        name = sound.export("test.wav",format="wav")
        sr, audio = read(name.name)
        assert(sr == sampling_rate)
        try:
            s_t, e_t = end_voice_clipping_by_Thr(audio, thr=100)
        except IndexError:
            continue
        audio = audio[s_t:e_t]
        if audio.shape[0] > int(sr * 0.02):
            features = MFCCextraction(audio,sr,winlen=0.02,winstep=0.01,numcep=19,nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.96,ceplifter=22,NORM=None,Delta = 'Yes',K=3)
            I += 1
        else:
            continue
        array_vect.append(features)
    f.create_array(f.root, "dimension", numpy.array([dim]))
    f.close()
    print(f"{I} number of files are processed and extracted the features of those many voices and stored in {dest_file}.\n")
    return True

def some_mfcc_features_file(folder_name = "test_folder", dest_file="some_h_file", dim = 38, sampling_rate=32000, f_mean = None, f_std = None):
    if os.path.isdir(folder_name) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), folder_name)
    f = tables.open_file(dest_file, mode = 'w')
    I = 0
    ls_files = os.listdir(folder_name)
    for filename in ls_files:
        name_file = os.path.join(folder_name, filename)
        sound = AudioSegment.from_file(name_file,format="mp3")
        name = sound.export("test.wav",format="wav")
        sr, audio = read(name.name)
        assert(sr == sampling_rate)
        try:
            s_t, e_t = end_voice_clipping_by_Thr(audio, thr=100)
        except IndexError:
            continue
        audio = audio[s_t:e_t]
        if audio.shape[0] > int(sr * 0.02):
            features = MFCCextraction(audio,sr,winlen=0.02,winstep=0.01,numcep=19,nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.96,ceplifter=22,NORM=None,Delta = 'Yes',K=3)
            I += 1
        else:
            continue
        res = (features - f_mean ) /  f_std
        child_name = 'feat' + str(I)
        f.create_array(f.root, child_name, res)
    f.create_array(f.root,'dimension',numpy.array([dim]))
    f.create_array(f.root,'numberOfUtternance', numpy.array([I]))
    f.close()
    print(f"{I} number of files are processed, extracted and normalized with CMNV end results are stored in {dest_file}.\n")
#-------------------------------------------------------------------------------------------------------------------------------------------------

#codebook generation code

def intra_cluster_stats(centroids,data,dcI):
    """Given m number of centroids (arranged in rows), each centroid is a n dim vector
    data contains N number of n dim vector i.e data is N by n vector
    dcI contains the N numbers each number represents the closest centroid index (from 0 to m-1)
    this function computes root means error square, standard deviation of data labelled same and number of associated data points to each centroid
    """
    n_centroids = centroids.shape[0] # number of centroids
    N = data.shape[0] #represents number of data points
    if n_centroids <= 0:
        raise ValueError(f"Number of centroids ({n_centroids}) should be atleast 1 in intra_cluster_stats function.\n")
    if N != len(dcI):
        raise ValueError(f"Number of data vectors ({N}) != number of labelled index (dcI {len(dcI)}) in intra_cluster_stats.\n")
    N_dp = numpy.zeros((n_centroids,))
    Err = numpy.zeros((n_centroids,)) # in case no data point associated to the centroid, Err  will be zero
    Std = numpy.copy(centroids) # in case no data point associated to the centroid, Standard deviaiton will be same as the centroid vector
    for i in range(0,n_centroids):
        n_d = numpy.where(dcI==i)[0].shape[0]
        N_dp[i] = n_d
        if n_d > 0:
            Err[i] = numpy.sqrt(numpy.sum(numpy.square( data[numpy.where(dcI==i)] - centroids[i] )) / n_d ) # this is a scalar
            Std[i,:] = numpy.std( data[numpy.where(dcI==i)] - centroids[i], axis=0 ) # this is a vector of n dim
    return Err, Std, N_dp

def reduce_no_centroids(centroids, Err, Std, N_data, thr, dcI, features):
    """This function will remove the cluster centers from the centroid arrays, if the number of labelled data associated with respective cluster centers are below than 
    threshold value thr.
    accordingly resize the Err, and Std array
    """
    if centroids.shape[0] != N_data.shape[0]:
        raise ValueError(f"number of centroids {centroids.shape[0]} !=  N_data {N_data.shape[0]} in function reduce_no_centroids.\n")
    Index = (numpy.where(N_data<=thr)[0]) # finding the set of index where there are not enough number of datas
    for i in range(Index.shape[0]): #iteratively getting rid of features and their labels
        features = features[numpy.where(dcI != Index[i])]
        dcI = dcI[numpy.where(dcI != Index[i])]
    # only storing the centroids, err and Std which has enough number of data to represent
    centroids = centroids[numpy.where(N_data>thr)]
    Err = Err[numpy.where(N_data>thr)]
    Std = Std[numpy.where(N_data>thr)]
    N_data = N_data[numpy.where(N_data>thr)]

    # have to relabel the dcI, since we are getting rid of some centroids which has less data supports in the training features
    dcI = map_centroids_to_oldlabels(centroids, dcI)
    #there is something about features (data) left out here, either it is unique signature of a person or related to noise
    return centroids, Err, Std, N_data, features, dcI

def map_centroids_to_oldlabels(centroids, dcI):
    #the logic of re-arranging labels
    various_indices = numpy.ones((centroids.shape[0],)) # this array maps range of centroids to labels
    k = 0
    for j in range(centroids.shape[0]): #mapping new indices to the old indices
        while (numpy.where(dcI==k)[0].shape[0] == 0):
            k += 1
        various_indices[j] = k
        k += 1
    for j in range(centroids.shape[0]): #replacing the new indicies in place of old indices
        dcI[numpy.where(dcI == various_indices[j])]  = j
    return dcI

def alocate_new_centroids(centroids, Err, N_dp, Inc=2):
    """Given the centroids with Err (RMSE) and number of supporting data features
    allocate additional number of centroids to the exisiting centroids based on the STATS of Err and N_dp
    """
    n_centroids = centroids.shape[0]
    if Inc <= 0:
        raise ValueError(f"Nothing to do in alocate_new_centroids function, given value is {Inc} that should be atleast 1. \n")
    W_E = (numpy.abs(Err) * N_dp) / numpy.sum(numpy.abs(Err) * N_dp)
    alc_newcentroids = numpy.zeros((n_centroids,))
    alcnC = Inc
    while alcnC > 0:
        if alc_newcentroids[numpy.where(alc_newcentroids == 0)].shape[0] == 0: # this logic is a safety logic, but It wont happen when I checked once
            print(f"Could not add {Inc} number of centroids, returning the function with less number of required centroids\n")
            return alc_newcentroids.astype(int)
        xam = -1
        ma_i = 0
        #following logic to find the maximum error indices which does not already had new cetroids recommendation
        for i in range(0,n_centroids):
            if (xam < W_E[i]) and (alc_newcentroids[i] == 0):
                xam = W_E[i]
                ma_i = i
        #following logic is add the number of new centroids along with the existing one cluster center 
        dda = 1 + numpy.ceil(xam * Inc)
        # following logic added to avoid more number of clusters
        if dda > alcnC + 1:
            dda = alcnC + 1
        # adding the new number of centers to the index ma_i
        alc_newcentroids[ma_i] = dda
        alcnC -= (dda - 1)
        xam = -1
        ma_i = 0
        if alcnC <= 0:
            break
    return alc_newcentroids.astype(int)
def find_simple_spread_vectors(center, n_v, st_dev):
    """Given n dimentional center, requried n_v number of random vectors, and standard deviation vector centered around the centroids st_dev
    find n_v number of new n dimenstional centroids around the center with st_dev deviation within 3 sigma
    """
    n = center.shape[0]
    New_centroids = numpy.zeros((n_v, n))
    for i in range(0,n_v):
        Vect = numpy.random.randn(n)
        Vect = Vect / numpy.sqrt(numpy.sum(numpy.square(Vect)))
        New_centroids[i] = center + Vect * st_dev * numpy.random.rand() * 3
    return New_centroids

def find_max_spread_vectors(center, n_v, st_dev):
    """Given n dimentional center, required n_v number of well spread vectors, with standard deviations st_dev
    consider to choose always n_v less than the dimention of the center vector
    """
    n = center.shape[0] # got the dim of the center
    if n_v > n:
        raise ValueError(f"find_rand_spread_vectors function is not a sophisticated one, can't handle n_v(={n_v}) > n(={n}) senarios\n")
    Vect = numpy.random.randn(n)
    Vect = Vect / numpy.sqrt(numpy.sum(numpy.square(Vect))) # found one random vector in n-dimensional space
    New_centroids = numpy.zeros((n_v, n))
    New_centroids[0] = center + Vect * st_dev *  numpy.random.rand()
    for i in range(0,n_v-1):
        new = Vect
        new[i] = -1 * new[i]
        New_centroids[i+1] = center + new * st_dev * (0.2 + numpy.random.rand() * 2.8)
    return New_centroids

def centroids_distance_with_1sigma_circle(centroids, Std):
    return numpy.sqrt(numpy.sum(numpy.square(centroids),axis=1)), numpy.sqrt(numpy.sum(numpy.square(Std),axis=1))

def dissimilarity_measure_STDBASED(data,centroids,Std,rStd=numpy.zeros((0,)),order=2,dsig=1):
    """ when using this function, consider to have good number of datas and also make sure the centeroids are calculated with good number of features
    order is the parameter deciding how well the scaling of  dissmiliarity should be
    """
    if (data.shape[1] != centroids.shape[1]):
        raise ValueError(f"Mismatch in the dimention of data shape {data.shape} and centeroids shape {centeroids.shape[1]}\n")
    dataLabel = label_data_set(data,centroids) # classify the data based on pre trained clusters of centers
    if (rStd.shape[0] == 0):
        _, rStd = centroids_distance_with_1sigma_circle(centroids, Std)
        rStd = (dsig * rStd) ** order # this dsig number decides the cut-off rate (first order effect) on the ratio data distance to the std
    SIM = 0.0
    nd = 0
    #Following function could have expressed interms of multi thread to reduce the real time computation
    for i in range(data.shape[0]):
        """Note if order is not one than
        dissimilarity is not linear but non-linear w.r.t distance
        """
        if rStd[dataLabel[i]] != 0:
            SIM += ( numpy.sum(numpy.square(centroids[dataLabel[i]] - data[i])) ** (order / 2) ) / rStd[dataLabel[i]]
            nd += 1
    if nd == 0:
        res = numpy.inf
    else:
        res = SIM / nd
    return res # SIM / data.shape[0]

def find_clusters(features,n_clusters=8,n_init=10,no_iter=300,covg=0.0001,algo="lloyd",init="k-means++"):
    VQCodes_cluster = KMeans(n_clusters=n_clusters,init=init,n_init=n_init,max_iter=no_iter,tol=covg,random_state=None,algorithm=algo)
    VQCodes_cluster.fit(features)
    centroids = VQCodes_cluster.cluster_centers_
    dcI = VQCodes_cluster.labels_
    return centroids, dcI

def label_data_set(data, centeroids):
    """I should have written myself this function instead of using KMeans from sklearn, will do later
    Mainly to express the function computation in terms of multi thread
    """
    codebook = KMeans(n_clusters=centeroids.shape[0],init=centeroids,n_init=1,max_iter=2)
    codebook.fit(centeroids)
    data_label = codebook.predict(data)
    return data_label

def VQCodeBook_exh(features,codebookSize=128,n_init=7,no_iter=300,covg=0.001,algo="lloyd",STEPS=numpy.array([2,4,16,32,64,128])):
    """For the given features of N rows of n-dim vectors (either mfcc, lpcc or PLP or what ever the feature extraction)
    find the codebookSize of clusters (a typical K mean stuff) creates a signature for the given set of features

    default codebookSize = 128 i.e number of clusters
    default maximum number of iterations (no_iter) consider to update the clusters
    with early stoping mechanism of convergence stats called as  convergence tolerance with default valule 0.001
    and algo type lloyd other option will be elkan (give these names as strings) this to submit into the sklearn libraries

    this function should return the codebookSize of centroids and also the respective standard deviation of the features w.r.t centroids
    """
    N = features.shape[0]
    n = features.shape[1]
    if codebookSize < 2:
        print(f"If you want {codebookSize} number of code book, calculate your self instead of calling VQCodeBook_exh function\n")
        return features[0], numpy.ones((n,))

    # make sure that STEPS dont have a number which is more than codebookSize number
    STEPS[numpy.where(STEPS > codebookSize)] = codebookSize
    # make sure that STEPS has a maximum number which should be the codebookSize number
    if STEPS[STEPS.shape[0]-1] < codebookSize:
        STEPS = numpy.append(STEPS,[codebookSize])

    if STEPS.shape[0] > 0:
        #calling Kmeans cluster from sklearn python library
        centroids, dcI = find_clusters(features,n_clusters=STEPS[0],n_init=7,no_iter=no_iter,covg=covg,algo=algo)
    else:
        # if the STEPS having negative value, then do brute force direct Kmeans clustering instead of step wise
        centroids, dcI = find_clusters(features,n_clusters=codebookSize,n_init=7,no_iter=no_iter,covg=covg,algo=algo)
        STEPS = numpy.array([codebookSize])

    #calcualte the RMS of each cluster centers around the features labelled as the same cluster
    #and standardar deviation of each cluster centers around the features labelled as the same cluster
    Err, Std, N_data = intra_cluster_stats(centroids,features,dcI)

    for s in range(1,STEPS.shape[0]):
        no_newc = STEPS[s] - centroids.shape[0]
        if no_newc > 0:
            alc_newcentroids = alocate_new_centroids(centroids, Err, N_data, Inc=no_newc)
            newcentroids = numpy.zeros((numpy.sum(alc_newcentroids),n))
            oldcentroids = centroids[numpy.where(alc_newcentroids==0)]
            si = 0
            for i in range(0,alc_newcentroids.shape[0]):
                if alc_newcentroids[i] > 0:
                    if alc_newcentroids[i] <= n:
                        newcentroids[si:si+alc_newcentroids[i],:] = find_max_spread_vectors(centroids[i], alc_newcentroids[i], Std[i])
                    else:
                        newcentroids[si:si+alc_newcentroids[i],:] = find_simple_spread_vectors(centroids[i], alc_newcentroids[i], Std[i])
                    si += alc_newcentroids[i]
            centroids = numpy.vstack((oldcentroids, newcentroids))
            centroids, dcI = find_clusters(features,n_clusters=STEPS[s],n_init=1,no_iter=no_iter,covg=covg,algo=algo,init=centroids)
            Err, Std, N_data = intra_cluster_stats(centroids,features,dcI)
        if STEPS[s] >= codebookSize:
            break
    return centroids,Std

#upto this point, all the above functions are tested
def generate_codebookVQ(source_file, dest_file, n_comp):
    if os.path.isfile(source_file) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source_file)
    if n_comp <= 1:
        raise ValueError(f"No need to run this program to generate VQ code book with size of VQ {n_comp}.\n")

    f = tables.open_file(source_file,mode="r")
    features = f.root.Norm_features[:,:]
    f.close()
    centroids, deviation = VQCodeBook_exh(features,codebookSize=n_comp,n_init=5,no_iter=300,covg=0.001,algo="lloyd",STEPS=numpy.array([2,4,16,32,64,128, 256, 512, 1024, n_comp]))
    f = tables.open_file(dest_file,'w')
    f.create_array(f.root, 'VQ', centroids)
    f.create_array(f.root, 'Dev', deviation)
    f.close()
    return True

#-----------------------------------------------------------------------------------------------------------------------
#gmm generation related code
# computing the log probability of given features with respect to  means and precision matrix
def estimateLogProb(features, means, precision):
    nC, n_features = means.shape
    n_samples = features.shape[0]
    log_det = numpy.sum(numpy.log(precision,dtype='float64'), axis=1,dtype='float64')
    log_prob = (
            numpy.sum((means**2 * precision), axis=1,dtype='float64')
            - 2.0 * numpy.dot(features, (means * precision).T)
            + numpy.dot(features**2, precision.T)
        )
    return -0.5 * (n_features * numpy.log(2.0 * numpy.pi,dtype='float64') + log_prob) + 0.5 * log_det

# computing log weighted probability
def logWeightedprob(log_prob, log_weights):
    weighted_log_prob = log_prob + log_weights
    log_norm = logsumexp(weighted_log_prob,axis=1)
    log_resp = weighted_log_prob - log_norm[:,numpy.newaxis]
    return log_resp, numpy.mean(log_norm,dtype='float64')

#estimating the respective parameter 
def estimateGaussianParameters(features, Q):
    nk = Q.sum(axis=0,dtype='float64') + 10 * numpy.finfo(Q.dtype).eps
    means = numpy.dot(Q.T, features) / nk[:, numpy.newaxis]
    avg_X2 = numpy.dot(Q.T, features * features) / nk[:, numpy.newaxis]
    avg_means2 = means**2
    avg_X_means = means * numpy.dot(Q.T, features) / nk[:, numpy.newaxis]
    covar = avg_X2 - 2 * avg_X_means + avg_means2 + 1e-7
    weights = nk / Q.shape[0]
    return weights, means, covar

#EM step in incremetal fashion
def gmmWorkers(queue,g, batch_size, nCmp,nIter,iWeights,iMean,iPrec,sW,sM,sC):
    N_feature_vector = g.root.Norm_features.shape[0]
    randIndex = random.sample(range(0, N_feature_vector - batch_size - 1), 1)[0]
    features = g.root.Norm_features[randIndex : randIndex + batch_size,:]
    dim = features.shape[1]
    nameWeights = shared_memory.SharedMemory(name=sW.name)
    arrayWeights = numpy.ndarray((nCmp,1), dtype=numpy.float64, buffer=nameWeights.buf)
    nameMeans = shared_memory.SharedMemory(name=sM.name)
    arrayMeans = numpy.ndarray((nCmp,dim), dtype=numpy.float64, buffer=nameMeans.buf)
    nameCovar = shared_memory.SharedMemory(name=sC.name)
    arrayCovar = numpy.ndarray((nCmp,dim), dtype=numpy.float64, buffer=nameCovar.buf)

    weights = iWeights
    means = iMean
    covar = (1 / iPrec) + 1e-6
    lower_bound = -numpy.inf # initializing the zeros lower bound
    for i in range(nIter):
        previous_lower_bound = lower_bound
        prec = (1 / covar)
        #Estimation Step
        logProb = estimateLogProb(features, means, prec)
        log_resp, lower_bound = logWeightedprob(logProb, numpy.log(weights,dtype='float64') )
        #Maximization step
        weights, mean, covar = estimateGaussianParameters(features, numpy.exp(log_resp, dtype='float64'))
        weights = weights / numpy.sum(weights,dtype='float64')
        change = numpy.abs((previous_lower_bound - lower_bound),dtype='float64')
        covar = covar + 1e-6
        if change < 0.001:
            break

    arrayWeights[:] = weights.reshape((nCmp,1))
    nameWeights.close()
    arrayMeans[:] = means
    nameMeans.close()
    arrayCovar[:] = covar
    nameCovar.close()
    print("In my processor, I done the job\n")

    """To represent this thread done is job to the called function, appending a int 1 in the queue
    """
    queue.put(int(1))
    return

#initializing the weights from vq code book
def _initializeWeights(NC,vqCentroids,g):
    weight_init = numpy.zeros((NC,),dtype='float64')
    features = g.root.Norm_features[:,:]
    codebook = KMeans(n_clusters=NC,init=vqCentroids,n_init=1,max_iter=2)
    codebook.fit(vqCentroids)
    dataLabel = codebook.predict(features)
    for c in range(NC):
        weight_init[c] = len(numpy.where(dataLabel == c)[0]) / dataLabel.shape[0]
    weight_init = weight_init / numpy.sum(weight_init, dtype='float64')
    print("Weights are initialized.\n")
    return weight_init

def _sharedMemoryArray(n_rows, n_cols):
    a = numpy.ones(shape=(n_rows,n_cols),dtype=numpy.float64)
    shmName = shared_memory.SharedMemory(create=True, size=a.nbytes)
    arrayShm = numpy.ndarray(shape=(n_rows,n_cols),dtype=numpy.float64,buffer=shmName.buf)
    arrayShm[:] = a[:]
    return shmName, arrayShm

def btStrpGMMParameters(
        convg=0.000001, # convergence limit for model parameters
        inc_iter = 50, # number of iteration for the repeated parameter updates procedure
        no_iter = 100, # batch iteration
        no_batch=8, # number of batch, limit this value based on number of threads you have
        batch_size = 50000, # number of features vector in each batch
        feature_file_name='someNormFeature_h5_file_path',
        vqcodeBook_centroids='someVQcodebook_h5_file_path',
        ):
    if os.path.isfile(vqcodeBook_centroids) == False:
        raise ValueError(f"There is no file called {vqcodeBook_centroids}.\n")
    """Following segment of codes load the centroids and covar from trained VQcodebook
    """
    f = tables.open_file(vqcodeBook_centroids, mode='r')
    NC, dim = f.root.VQ.shape[0], f.root.VQ.shape[1]
    newMean = f.root.VQ[:,:]
    stdDev = f.root.Dev[:,:]
    f.close()
    stdDev[stdDev < 1e-6] =  1e-6
    newCovar = numpy.square(stdDev,dtype='float64')
    del(stdDev)

    #for the parameter convergence
    CONVG = convg * (NC + 2.0 * dim * NC)
    """checking the normed feature files to train gmm parameters
        also check the batch size and batch number with respect to the number of features we have for training
    """
    if os.path.isfile(feature_file_name) == False:
        raise ValueError(f"There is no file called {feature_file_name}.\n")
    g = tables.open_file(feature_file_name,mode='r')
    N_feature_vector = g.root.Norm_features.shape[0]
    n_features = g.root.Norm_features.shape[1]
    if n_features != dim:
        raise ValueError(f"There is mismatch in dimension of vqcodebook ({dim}) and dimension of Normed features ({n_features}).\n")
    if N_feature_vector < int(no_batch * batch_size):
        raise ValueError(f"There are not enough number of features to train according to number of batch_size and no_batch {batch_size}, {no_batch}.\n")

    """ Initializing the weights for gmm parameters
    """
    weights = _initializeWeights(NC,newMean,g)
    #creating a file name to store the intermediate gmm parameters to save the computation work
    intermediate_save_file = vqcodeBook_centroids[:-3] + "_intermediate_gmmResults.h5"
    #Initializing shared weights array here
    shmWeights = list()
    arWeights = list()
    shmMeans = list()
    arMeans = list()
    shmCovar = list()
    arCovar = list()
    #creating shared memory to get weights, covar, means for the increment update
    for i in range(no_batch):
        sName, shmAr = _sharedMemoryArray(n_rows = NC, n_cols = 1)
        shmWeights.append(sName)
        arWeights.append(shmAr)
        #print(f"{i+1} th batch weights name is {sName.name}.\n")

        sName, shmAr = _sharedMemoryArray(n_rows = NC, n_cols = dim)
        shmMeans.append(sName)
        arMeans.append(shmAr)

        sName, shmAr = _sharedMemoryArray(n_rows = NC, n_cols = dim)
        shmCovar.append(sName)
        arCovar.append(shmAr)
    print("Initialized shared memory between threads and main function.\n")

    stTime = time.perf_counter()
    for l in range(no_iter):
        old_weights = weights
        old_mean = newMean
        old_covar = newCovar
        newCovar[newCovar < 1e-6] = 1e-6
        precision = 1.0 / newCovar
        queue = Queue()
        processors = [
                Process(target=gmmWorkers,args=(queue, g, batch_size,
                    NC,inc_iter,weights,newMean,precision,shmWeights[b],shmMeans[b],shmCovar[b])) for b in range(no_batch)
                ]
        for p in processors:
            p.start()
        ASUM = sum([queue.get() for i in range(no_batch)])
        assert(ASUM == no_batch)
        for p in processors: # meeting all threads to a common point in this main function
            p.join()
        if l == 0:
            con = 1.0 / no_batch
            weights = numpy.zeros(old_weights.shape, dtype='float64')
            newMean = numpy.zeros(old_mean.shape, dtype='float64')
            newCovar = numpy.zeros(old_covar.shape, dtype='float64')
        else:
            con = 0.2 / no_batch # this special number indicate the typical percentage of total data loaded in single iteration
            weights = 0.8 * old_weights
            newMean = 0.8 * old_mean
            newCovar = 0.8 * old_covar
        for b in range(no_batch):
            weights += ( con * arWeights[b][:,0])
            newMean += ( con * arMeans[b])
            newCovar += ( con * arCovar[b])
        weights = weights / weights.sum(dtype='float64')
        c_converg = numpy.sum(numpy.abs((weights-old_weights),dtype='float64')) + numpy.sum(numpy.sum(numpy.abs((newMean-old_mean),dtype='float64')))
        c_converg = c_converg + numpy.sum(numpy.sum(numpy.abs((newCovar-old_covar),dtype='float64')))
        print(f"At {l+1} iteration, convergence value is {c_converg}.\n")
        if c_converg < CONVG:
            break
        if (l % 10) == 0:
            stTime = time.perf_counter() - stTime
            print(f"At {l+1} iteration, convergence value: {c_converg} (expected convergence {CONVG}) and tooks {stTime / 60} minutes.\n")
            print(f"We are saving the intermediate paratmeters in a file: {intermediate_save_file}.\n")
            svIntermediateGMM = tables.open_file(intermediate_save_file,'w')
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Weights', weights)
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Means', newMean)
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Covar', newCovar)
            svIntermediateGMM.create_array(svIntermediateGMM.root,'itNumber', numpy.array([l+1],dtype='int32'))
            svIntermediateGMM.close()
            stTime = time.perf_counter()
    if c_converg > CONVG:
        print(f"Warning: In bootStrap GMM update, parameters are not converged, current convergence value: {c_converg} expected convergence value {CONVG}, consider to do more iteration.\n")
    g.close()
    for b in range(no_batch):
        shmWeights[b].close()
        shmWeights[b].unlink()
        shmMeans[b].close()
        shmMeans[b].unlink()
        shmCovar[b].close()
        shmCovar[b].unlink()
    return weights, newMean, newCovar


def genUBMGMMBootStrap(file_path = 'Some_Norm_features_hfile', vqpath = 'some_h5_vqcodebook_file', res_path = 'gmm_ubm_parameters'):
    
    intermediateFileName = vqpath[:-3] + "_intermediate_gmmResults.h5"
    print(f"Make sure that you dont have a file name (path) {intermediateFileName}, regularly the intermedate gmm parameters will be stored in the mentioned file path.\n")
    weights, means, covar = btStrpGMMParameters(convg=0.000001,inc_iter = 100,no_iter = 80,
            no_batch=4,batch_size = 600000,feature_file_name=file_path,
            vqcodeBook_centroids=vqpath)
    print(f"Succesfully trainined UBM (diagonal covar) model from the features file {file_path}, will store results in {res_path}.\n")
    f = tables.open_file(res_path,'w')
    f.create_array(f.root, 'Weights', weights)
    f.create_array(f.root, 'Means', means)
    f.create_array(f.root, 'Covar', covar)
    f.close()
    return weights,means,covar
#----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    sampling_rate = 32000
    dim = 38
    src_folder = "cv-corpus-11.0-delta-2022-09-21/fr/clips"
    french_features_file = "cv-corpus-11.0-delta-2022-09-21/fr/MFCCFeaturesdim38.h5"
    filenameMean = 'cv-corpus-11.0-delta-2022-09-21/fr/frenchFeatureMean.npy'
    filenameSTD = 'cv-corpus-11.0-delta-2022-09-21/fr/frenchFeatureStd.npy'
    french_features_Norm_file = "cv-corpus-11.0-delta-2022-09-21/fr/MFCCNormFeaturesdim38.h5"
    d_file1024 = 'cv-corpus-11.0-delta-2022-09-21/fr/VQCodeBook1024French.h5'
    french_gmm1024 = "cv-corpus-11.0-delta-2022-09-21/fr/frenchGMM1024.h5"
    specimen_norm_featuresFrench = "cv-corpus-11.0-delta-2022-09-21/fr/specimenFrenchMFCCNormFeatures.h5" 
    
    print("Extracting the feature of given file path {src_folder}.\n")
    s_time = time.perf_counter()
    global_mfcc_features_file(folder_name=src_folder, dest_file=french_features_file, dim=dim, sampling_rate=sampling_rate)
    s_time = time.perf_counter() - s_time
    print(f"In the extraction of features from the folder {src_folder} tooks {s_time / 60} minutes.\n")
    
    print(f"Computing the first and second order stats of the features {french_features_file}.\n")
    s_time = time.perf_counter()
    f_mean, f_std = compute_STATS_feature(file_path = french_features_file)
    numpy.save(filenameMean, f_mean)
    numpy.save(filenameSTD, f_std)
    s_time = time.perf_counter() - s_time
    print(f"Computing the stats tooks {s_time} seconds.\n")
    
    print("Computing the normalized features of the features.\n")
    s_time = time.perf_counter()
    normalize_features_file(f_mean = f_mean, source_file = french_features_file, dest_file = french_features_Norm_file, NORM = 'CMNV', f_std = f_std)
    s_time = time.perf_counter() - s_time
    print(f"Computing normalized features tooks {s_time / 60} minutes.\n")

    print("Extracting MFCC specimen norm features for i-vector training.\n")
    s_time = time.perf_counter()
    f_mean = numpy.load(filenameMean)
    f_std = numpy.load(filenameSTD)
    some_mfcc_features_file(folder_name = src_folder, dest_file=specimen_norm_featuresFrench, dim = dim, sampling_rate=sampling_rate, f_mean = f_mean, f_std = f_std)
    s_time = time.perf_counter() - s_time
    print(f"In extracting the specimen feature sample, program tooks {s_time / 60} minutes.\n\n")

    print(f"\n\nComputing Kmean cluster for the given feautres file.\n")
    s_time = time.perf_counter()
    generate_codebookVQ(source_file = french_features_Norm_file, dest_file = d_file1024, n_comp = 1024)
    s_time = time.perf_counter() - s_time
    print(f"In VQcode book generation 1024 tooks {s_time / 60} minutes.\n")


    #now we will move on to generating the gmm components
    print("\n\nSystem starts to compute gmm parameters")
    s_time = time.perf_counter()
    genUBMGMMBootStrap(file_path = french_features_Norm_file, vqpath = d_file1024, res_path = french_gmm1024)
    s_time = time.perf_counter() - s_time
    print(f"IN GMM generation of 1024 components, it tooks {s_time / 60} minutes.\n")
    
