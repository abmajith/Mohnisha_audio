import numpy
import sys
sys.path.append('../')
from frontend_processing import *
from scipy.io.wavfile import read
import time
from mixture_model import *
import os
from codebook_generation  import *


def Train(subfolder = '/home/abdul/Projects/Mohnisha_audio/datasets/train_voices'):
    audio_files = os.listdir(subfolder)
    folder = subfolder
    voice_sample, sr = loadWAVFiles(audio_files, folder, CLIP = True)
    #finding lpcc of voice
    mf_voice_feat = numpy.zeros((0,))
    for voice in voice_sample:
        #features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
        #features = PLPextraction(voice,sr,NORM ='CMVN')
        if (mf_voice_feat.shape[0] == 0):
            mf_voice_feat = features
            #gmModel = create_andtrain_GMMmodel(features,cluster_comp=200,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7)
            #n_components, model_weights, model_means, model_covar = get_GMM_parameters(gmModel)
        else:
            mf_voice_feat = numpy.vstack((mf_voice_feat,features))
            #model_weights, model_means, model_covar = MAP_update_GMM_parameters(features,n_components,model_weights,model_means,model_covar,r=16)
        #centroids,Std = VQCodeBook_exh(features,codebookSize=200,n_init=7,no_iter=300,covg=0.001,algo="lloyd",STEPS=numpy.array([2,4,16,32,64,128]))
        gmModel = create_andtrain_GMMmodel(mf_voice_feat,cluster_comp=100,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7)
        #n_components = centroids.shape[0]
        #model_weights = numpy.ones((n_components,))
        #model_means = centroids
        #model_covar = numpy.square(Std)
        n_components, model_weights, model_means, model_covar = get_GMM_parameters(gmModel)
    return model_weights, model_means, model_covar, n_components
def Test(model_weights, model_means, model_covar, n_components):
    same_voice_sample, sr = loadWAVFiles(audio_files=os.listdir('/home/abdul/Projects/Mohnisha_audio/datasets/test_voices/same'),
            foldername = '/home/abdul/Projects/Mohnisha_audio/datasets/test_voices/same', CLIP = True)
    diff_voice_sample, sr = loadWAVFiles(audio_files=os.listdir('/home/abdul/Projects/Mohnisha_audio/datasets/test_voices/diff'),
            foldername = '/home/abdul/Projects/Mohnisha_audio/datasets/test_voices/diff', CLIP = True)
    same_voice_log_score = list()
    diff_voice_log_score = list()
    for voice in same_voice_sample:
        #features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
        #features = PLPextraction(voice,sr,NORM ='CMVN')
        arr_log = get_log_likelihood(features=features,no_compn=n_components,model_weights=model_weights,model_means=model_means,model_covar=model_covar)
        same_voice_log_score.append(numpy.average(arr_log))
    for voice in diff_voice_sample:
        #features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
        #features = PLPextraction(voice,sr,NORM ='CMVN')
        arr_log = get_log_likelihood(features=features,no_compn=n_components,model_weights=model_weights,model_means=model_means,model_covar=model_covar)
        diff_voice_log_score.append(numpy.average(arr_log))
    return same_voice_log_score, diff_voice_log_score

def Train_Test():
    st = time.perf_counter()
    model_weights, model_means, model_covar, n_components = Train('/home/abdul/Projects/Mohnisha_audio/datasets/train_voices')
    st = time.perf_counter() - st
    st1 = time.perf_counter()
    s_v_log_score, d_v_log_score = Test(model_weights, model_means, model_covar, n_components)
    st1 = time.perf_counter() - st1
    print(f"The train test took {st,  st1} seconds\n")
    print("The log likelihood score for the same voices are \n ")
    for v in s_v_log_score:
        print(f"\t{v}\n")
    print("\nThe log likelihood score for the diff voices are \n ")
    for v in d_v_log_score:
        print(f"\t{v}\n")
    print("\n")
    return True

if __name__ == "__main__":
    Train_Test()


