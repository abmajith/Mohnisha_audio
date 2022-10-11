import numpy
import sys
sys.path.append('../')
from frontend_processing import *
from scipy.io.wavfile import read
import time
from mixture_model import *
import os
from codebook_generation  import *

"""

features = numpy.array(([1,2,0.5],[3,2,1.5],[1,2.5,1],[3,2.1,1],[3,2.1,1]))
#features = numpy.random.randn(18).reshape(6,3)
ubm = create_andtrain_GMMmodel(features,cluster_comp=2,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7)
assert(ubm.covariances_.shape[0] == 2)
assert(ubm.covariances_.shape[1] == 3)

n_components, weights, means, covar = get_GMM_parameters(ubm)
assert(n_components == 2)
assert(weights.shape[0] == 2 and means.shape[0] == ubm.means_.shape[0] and means.shape[1] == ubm.means_.shape[1])
print("The weights of the componets are\n")
for i in range(weights.shape[0]):
    print(f"{weights[i]}\t")
print("\n")

data = numpy.array(([1,1,1],[3,2,1],[3,2,1]))
prob = get_proba(data, n_components, means, covar)
assert(prob.shape[0] == data.shape[0] and prob.shape[1] == n_components)
print(f"The prob feat dimenstions are {prob.shape[0]}, {prob.shape[1]}\n")
for i in range(prob.shape[0]):
    print(f"{prob[i]}\n")


log_data = get_log_likelihood(data,n_components,weights,means,covar)
assert(log_data.shape[0] == data.shape[0])
for i in range(log_data.shape[0]):
    print(f"{log_data[i]}\t")
print("\n")


new_features = numpy.array(([.04,0.2,0.1],[1,2,3.1],[2,1,1.4],[2,3,1]))
weights, mean, covar = MAP_update_GMM_parameters(new_features,no_compn=n_components,model_weights=weights,model_means=means,model_covar=covar,r=16)

print(f"weights of the components {weights}\t mean of the components {mean} and covariance of the components {covar}\n")
"""


st = time.perf_counter()
sr, voice = read('test.wav')
features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)

gmModel = create_andtrain_GMMmodel(features,cluster_comp=200,covar_type = 'diag',convg=0.001,no_iter=200,n_init=7)
st = time.perf_counter() - st
print(st)

n_components, weights, means, covar = get_GMM_parameters(gmModel)
"""
new_weights, new_means, new_covar = MAP_update_GMM_parameters(features,no_compn = n_components, model_weights = weights, model_means = means ,model_covar = covar,r=16)
print(f"The gmm based parametes are {weights}, {means}, {covar}\n")
print(f"The map based parameters are {new_weights}, {means}, {covar}\n")
"""

prob = get_proba(features, no_compn = n_components, model_means=means, model_covar=covar)
w_prob = get_weighted_prob(prob, weights)
"""
for i in range(40):
    print(f"{numpy.max(prob[i]), numpy.argmax(prob[i]), numpy.max(w_prob[i])}, {numpy.argmax(w_prob[i])}\n")
arr = get_log_likelihood(features,n_components,weights,means,covar)
print(f"{numpy.average(arr)}")
"""
#print(f"{means} \n {covar}\n")
stats = numpy.sum(w_prob, axis=0)
#print(f"{stats}\n")

n_stats = stats / features.shape[0]
#print(f"{weights}\n")
alpha = (stats / (stats + 16)) * stats / features.shape[0]  + ( 1 - (stats / (stats + 16)) ) * weights
print(f"{alpha}\n")
s = alpha.sum(dtype='float64')
beta = alpha / s
print(f"{beta.sum(dtype='float64')}")
#assert(beta.sum() == 1.0)
"""
def Train(subfolder = '/home/abdul/Projects/Mohnisha_audio/datasets/train_voices'):
    audio_files = os.listdir(subfolder)
    folder = subfolder
    voice_sample, sr = loadWAVFiles(audio_files, folder, CLIP = True)
    #finding lpcc of voice
    mf_voice_feat = numpy.zeros((0,))
    for voice in voice_sample:
        features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        #features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
        #features = PLPextraction(voice,sr,NORM ='CMVN')
        if (mf_voice_feat.shape[0] == 0):
            mf_voice_feat = features
            #gmModel = create_andtrain_GMMmodel(features,cluster_comp=200,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7)
            #n_components, model_weights, model_means, model_covar = get_GMM_parameters(gmModel)
        else:
            mf_voice_feat = numpy.vstack((mf_voice_feat,features))
            #model_weights, model_means, model_covar = MAP_update_GMM_parameters(features,n_components,model_weights,model_means,model_covar,r=16)
        gmModel = create_andtrain_GMMmodel(mf_voice_feat,cluster_comp=200,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7)
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
        features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        #features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
        #features = PLPextraction(voice,sr,NORM ='CMVN')
        arr_log = get_log_likelihood(features=features,no_compn=n_components,model_weights=model_weights,model_means=model_means,model_covar=model_covar)
        same_voice_log_score.append(numpy.average(arr_log))
    for voice in diff_voice_sample:
        features = LPCCextraction(voice,sr,winlen=0.025,hoplen=0.01,Q=18,p=12,emph_coeff=0.95,NORM='CMVN',Delta='Yes',K=3)
        #features = MFCCextraction(voice,sr,NORM='CMVN',Delta = 'Yes',K=3)
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
"""
