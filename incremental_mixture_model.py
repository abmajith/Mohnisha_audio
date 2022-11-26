import numpy
import tables
import os
import errno
import random
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from multiprocessing import Process, Queue, shared_memory, Pool
import time
from scipy.special import logsumexp

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
    #to avoid memory leakage
    
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
        if c_converg < CONVG:
            break
        if (l+1) % 10 == 0:
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

#this is the same kind of training, but instead of using our function, calling external functions
def incExtgmmtrain(
        convg=0.001, # convergence limit for model parameters
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

    """checking the normed feature files to train gmm parameters
        also check the batch size and batch number with respect to the number of features we have for training
    """
    if os.path.isfile(feature_file_name) == False:
        raise ValueError(f"There is no file called {feature_file_name}.\n")
    g = tables.open_file(feature_file_name,mode='r')
    N_feature_vector = g.root.Norm_features.shape[0]
    n_features = g.root.Norm_features.shape[1]
    if n_features != dim:
        raise ValueError(f"There is mismatch in dimension of vqcodebook {dim} and dimension of Normed features {n_features}.\n")
    if N_feature_vector < int(no_batch * batch_size):
        raise ValueError(f"There are not enough number of features to train according to the number of batch_size and no_batch {batch_size}, {no_batch}.\n")
    
    #for the parameter convergence
    CONVG = convg * (NC + 2.0 * dim * NC)
    """ Initializing the weights for gmm parameters
    """
    weights = _initializeWeights(NC,newMean,g)
    
    intermediate_save_file = vqcodeBook_centroids[:-3] + "_intermediate_gmmResults.h5"
    stTime = time.perf_counter()
    for l in range(no_iter):
        old_weights = weights
        old_mean = newMean
        old_covar = newCovar
        newCovar[newCovar < 1e-6] = 1e-6
        precision = 1.0 / newCovar
        arWeights = list()
        arMeans = list()
        arCovar = list()
        for b in range(no_batch):
            randIndex = random.sample(range(0, N_feature_vector - batch_size - 1), 1)[0]
            features = g.root.Norm_features[randIndex : randIndex + batch_size,:]
            gmModel = GaussianMixture(n_components=NC,covariance_type='diag',tol=0.001,max_iter=inc_iter,weights_init=weights, means_init=newMean, precisions_init=precision)
            gmModel.fit(features)
            arWeights.append(gmModel.weights_)
            arMeans.append(gmModel.means_)
            arCovar.append(gmModel.covariances_)
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
            weights = ( con * arWeights[b])  + weights
            newMean = ( con * arMeans[b]) + newMean
            newCovar = ( con * arCovar[b]) + newCovar
        weights = weights / weights.sum(dtype='float64')
        c_converg = numpy.sum(numpy.abs((weights-old_weights),dtype='float64')) + numpy.sum(numpy.sum(numpy.abs((newMean-old_mean),dtype='float64')))
        c_converg = c_converg + numpy.sum(numpy.sum(numpy.abs((newCovar-old_covar),dtype='float64')))
        if c_converg < CONVG:
            break
        if (l+1) % 10 == 0:
            stTime = time.perf_counter() - stTime
            print(f"At {l+1} iteration, convergence value: {c_converg} and tooks {stTime / 60} minutes.\n")
            print(f"We are saving the intermediate paratmeters in a file: {intermediate_save_file}.\n")
            svIntermediateGMM = tables.open_file(intermediate_save_file,'w')
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Weights', weights)
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Means', newMean)
            svIntermediateGMM.create_array(svIntermediateGMM.root, 'Covar', newCovar)
            svIntermediateGMM.create_array(svIntermediateGMM.root,'itNumber', numpy.array([l+1],dtype='int32'))
            svIntermediateGMM.close()
            stTime = time.perf_counter()
    if c_converg > convg:
        print(f"Warning: In bootStrap GMM update, parameters are not converged, current convergence value: {c_converg}, consider to do more iteration.\n")
    g.close()
    return weights, newMean, newCovar

def genUBMGMMBootStrap(file_path = 'Some_Norm_features_hfile', vqpath = 'some_h5_vqcodebook_file', res_path = 'gmm_ubm_parameters'):

    intermediateFileName = vqpath[:-3] + "_intermediate_gmmResults.h5"
    print(f"Make sure that you dont have a file name (path) {intermediateFileName}, regularly the intermedate gmm parameters will be stored in the mentioned file path.\n")
    weights, means, covar = btStrpGMMParameters(convg=0.000001,inc_iter = 100,no_iter = 120,
            no_batch=4,batch_size = 175000,feature_file_name=file_path,
            vqcodeBook_centroids=vqpath)
    print(f"Succesfully trainined UBM (diagonal covar) model from the features file {file_path}, will store results in {res_path}.\n")
    f = tables.open_file(res_path,'w')
    f.create_array(f.root, 'Weights', weights)
    f.create_array(f.root, 'Means', means)
    f.create_array(f.root, 'Covar', covar)
    f.close()
    return weights,means,covar
#----------------------------------------------------------------------------------------------------------------------------------------------

