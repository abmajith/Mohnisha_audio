import numpy
import scipy
import os
import tables
import time
from multiprocessing import Process, Queue, shared_memory
from scipy.special import logsumexp

#------------------------------------------------------------------------
#defining some global variables so that repeated usage of parameters are avoided
_glbMinDivCond = False
_glbNC = 2
_glbNF = 2
_glbTVn_col = 2
_glbTVn_row = int(_glbNC * _glbNF)
_glbPrec = numpy.ones((_glbNC,_glbNF),dtype='float64')
_glbWeight = numpy.ones((_glbNC,),dtype='float64')
_glbLogWeight = numpy.zeros((_glbNC,),dtype='float64')
_glbMean = numpy.zeros((_glbNC,_glbNF),dtype='float64')
_glbSqrtPrec = numpy.ones((_glbNC,_glbNF),dtype='float64')
_glbCovar = numpy.ones((_glbNC,_glbNF),dtype='float64')
_glbLogDet = numpy.zeros((_glbNC,),dtype='float64')
_glbMeanPrecProdSum = numpy.zeros((_glbNC,),dtype='float64')
_glbMeanPrecProd = numpy.zeros((_glbNF,_glbNC),dtype='float64')
#-----------------------------------------------------------------------
def initializeSettings(gmm_file="Some_h5_gmmFile", n_col = 100, divCodn = False):
    if os.path.isfile(gmm_file) == False:
        raise ValueError(f"There is no file called {gmm_file}.\n")
    f = tables.open_file(gmm_file, mode='r')
    global _glbMinDivCond
    global _glbNC
    global _glbNF
    global _glbTVn_col
    global _glbTVn_row
    global _glbWeight
    global _glbLogWeight
    global _glbMean
    global _glbCovar
    global _glbPrec
    global _glbSqrtPrec
    global _glbLogDet
    global _glbMeanPrecProdSum
    global _glbMeanPrecProd

    _glbMinDivCond = divCodn
    _glbNC, _glbNF = f.root.Means.shape[0], f.root.Means.shape[1] # number of component and number of features
    _glbTVn_row = int(_glbNC * _glbNF)
    _glbTVn_col = n_col
    TV_mat = numpy.random.randn( _glbTVn_row, _glbTVn_col)
    _glbWeight = f.root.Weights
    _glbLogWeight = numpy.log(_glbWeight,dtype='float64')
    _glbMean = f.root.Means[:,:]
    _glbCovar = f.root.Covar[:,:] + 1e-8
    f.close()
    _glbPrec = 1.0 / _glbCovar
    _glbSqrtPrec = numpy.sqrt(_glbPrec,dtype='float64')
    _glbLogDet = numpy.sum(numpy.log(_glbSqrtPrec), axis=1)
    _glbMeanPrecProdSum = numpy.sum((_glbMean**2 * _glbPrec), axis=1)
    _glbMeanPrecProd = (_glbMean * _glbPrec).T
    for i in range(_glbNC):
        TV_mat[i * _glbNF : (i + 1) * _glbNF, :] = (_glbSqrtPrec[i,:] * TV_mat[i * _glbNF : (i + 1) * _glbNF, :].T).T
    return TV_mat

def statsZeroFirstOrder(features):
    n_samples = features.shape[0]
    log_prob = (
            _glbMeanPrecProdSum
            - 2.0 * numpy.dot(features, _glbMeanPrecProd)
            + numpy.dot(features**2, _glbPrec.T)
        )
    logProb = -0.5 * (_glbNF * numpy.log(2.0 * numpy.pi) + log_prob) + _glbLogDet
    logWeightProb = _glbLogWeight + logProb

    log_norm = logsumexp(logWeightProb,axis=1)
    log_resp = logWeightProb - log_norm[:,numpy.newaxis]

    prob = numpy.exp(log_resp,dtype='float64')
    zeroStat = prob.sum(axis=0,dtype='float64') + 10 * numpy.finfo(prob.dtype).eps
    firstStat = numpy.dot(prob.T, features)
    firstStat = firstStat - (_glbMean.T * zeroStat).T
    firstStat = firstStat * _glbSqrtPrec
    return zeroStat, firstStat
"""zeroStat is a vector of number of components
   firstStat is a matrix of number of components cross number of features
"""

def iParameterEstimation(TV_mat, zeroStat, firstStat):
    LUtternance = numpy.identity(_glbTVn_col,dtype='float64') # n_col cross n_col matrix, a fixed paratmer estimation for given stats of features
    for i in range(_glbNC):
        LUtternance += zeroStat[i] * numpy.matmul(TV_mat[i * _glbNF : (i + 1) * _glbNF,:].T, TV_mat[i * _glbNF : (i + 1) * _glbNF,:], dtype='float64')
    invLutt = numpy.linalg.inv(LUtternance)
    iUtternance = numpy.matmul( numpy.matmul(invLutt, TV_mat.T), numpy.matrix.flatten(firstStat), dtype='float64' )
    return iUtternance, invLutt #, LUtternance
"""iUtternance is a vector of size n_col
invLutt is n_col cross n_col matrix stores inverse covariance matrix for the number of components
LUtternance is n_col cross n_col matrix stores covariance matrix for the number of components
"""
def maximizationStep(lstIvecUtt, lstzeroStat, lstfirstStat, lstInvL):
    nUtt = lstfirstStat.shape[0]
    CLCT = numpy.outer(numpy.matrix.flatten(lstfirstStat[0,:,:]), lstIvecUtt[0,:]) #create (nC * nFeat) cross n_col matrix
    for j in range(1,lstIvecUtt.shape[0]):
        CLCT += numpy.outer(numpy.matrix.flatten(lstfirstStat[j,:,:]), lstIvecUtt[j,:])
    T = numpy.zeros((_glbTVn_row,_glbTVn_col),dtype='float64')
    A_com = numpy.zeros((nUtt,_glbTVn_col,_glbTVn_col),dtype='float64')
    for i in range(nUtt):
        A_com[i,:,:] = lstInvL[i,:,:] + numpy.outer(lstIvecUtt[i,:], lstIvecUtt[i,:])
    for i in range(_glbNC):
        A_c = numpy.zeros((_glbTVn_col,_glbTVn_col),dtype='float64')
        for j in range(nUtt):
            A_c += lstzeroStat[j,i] * A_com[j,:,:]
        T[i * _glbNF:(i+1) * _glbNF,:] = numpy.matmul(CLCT[i * _glbNF:(i+1) * _glbNF,:], numpy.linalg.inv(A_c))
    return T
#tested
def cosine_similarity_score(w_target, w_test):
    #computing cosine of angle between two given vectors
    score = numpy.dot(w_target, w_test) / ( numpy.sqrt(numpy.dot(w_target, w_target), dtype='float64') * numpy.sqrt(numpy.dot(w_test,w_test),dtype='float64') )
    return score

def minimizeDivergence(lstIvec, TV_mat):
    nUtt = lstIvec.shape[0]
    secondStat = numpy.cov(lstIvec.T, dtype='float64')
    uppCholeskyStar = sqrtSymMat(secondStat).T
    return numpy.matmul(TV_mat, uppCholeskyStar)
#------------------------------------------------------------------------------------------------------------------
#function for creating the shared memory blocks
def sharedMemoryMatrix():
    a = numpy.ones(shape=(_glbTVn_row,_glbTVn_col),dtype=numpy.float64)
    shmName = shared_memory.SharedMemory(create=True, size=a.nbytes)
    arrayShm = numpy.ndarray(shape=(_glbTVn_row,_glbTVn_col),dtype=numpy.float64,buffer=shmName.buf)
    arrayShm[:] = a[:]
    return shmName, arrayShm
#---------------------------------------------------------------------------------------
#couple of functions to make given matrix to a positive definite matrix
def isPD(B): #to check the given matrix is positive definite
    try:
        _ = numpy.linalg.cholesky(B)
        return True
    except numpy.linalg.LinAlgError:
        return False

def sqrtSymMat(Mat): # computing cholesky upper triangluar inverse matrix
    try:
        result = scipy.linalg.cholesky(Mat) # straight forward method
    except scipy.linalg.LinAlgError:
        # following methods to take precision mistake (even given matrix is PD) and compute cholesky decomposition accrd.
        re = (Mat + Mat.T) / 2.0
        _, s, V = numpy.linalg.svd(re)
        H = numpy.dot(V.T, numpy.dot(numpy.diag(s), V))
        A2 = (re + H) / 2
        A3 = (A2 + A2.T) / 2
        if isPD(A3):
            result = scipy.linalg.cholesky(A3)
        else: # even the symettrization not helping, that means given matrix is not actually positive definite
            spacing = numpy.spacing(numpy.linalg.norm(Mat))
            I = numpy.eye(Mat.shape[0])
            k = 1
            while not isPD(A3):
                mineig = numpy.min(numpy.real(numpy.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**2 + spacing)
                k += 1
            result = scipy.linalg.cholesky(A3)
    return result
#-----------------------------------------------------------------------------------------------

def TVWorkers(queue,g,no_utt,max_utt,nIter,iTVmat,sTV):
    if no_utt < max_utt:
        randIndex = random.sample(range(1, max_utt+1),no_utt)
        NUtt = no_utt
    else:
        randIndex = range(1,max_utt+1)
        NUtt = max_utt
    #following code do Stats of the features
    lstIvecUtt = numpy.zeros((NUtt,_glbTVn_col),dtype='float64')
    lstZeroStat = numpy.zeros((NUtt,_glbNC),dtype='float64')
    lstFirstStat = numpy.zeros((NUtt,_glbNC,_glbNF), dtype='float64')
    lstInvL = numpy.zeros((NUtt,_glbTVn_col,_glbTVn_col), dtype='float64')
    j = 0
    for i in randIndex:
        d_name = 'feat' + str(i)
        features = g.root[d_name][:]
        lstZeroStat[j,:], lstFirstStat[j,:,:] = statsZeroFirstOrder(features)
        j += 1
    del(randIndex)

    #now we will do estimation-maximization step in iterative way until it converges
    oldTV = iTV
    for i in range(nIter):
        for j in range(NUtt):
            lstIvecUtt[j,:], lstInvL[j,:,:] = iParameterEstimation(oldTV, lstZeroStat[j,:], lstFirstStat[j,:,:])
        newTV = maximizationStep(lstIvecUtt, lstZeroStat, lstFirstStat, lstInvL)
        if (_glbMinDivCond == True):
            newTV = minimizeDivergence(lstIvecUtt, newTV)
        n_convg = numpy.sum(numpy.sum(numpy.abs((newTV-oldTV),dtype='float64')))
        oldTV = newTV
        if n_convg < 0.001:
            break
    #now we will load the computed TV matrix in the shared memory space
    nameTV = shared_memory.SharedMemory(name=sTV.name)
    arrayTV = numpy.ndarray((_glbTVn_row,_glbTVn_col), dtype=numpy.float64, buffer=nameTV.buf)
    arrayTV[:] = newTV
    nameTV.close()
    queue.put(int(1))
    return

def btStrapTVMatrix(n_col = 150,
        convg=0.001, # convergence limit for model parameters
        inc_iter = 50, # number of iteration for the repeated parameter updates procedure
        no_iter = 100, # batch iteration
        no_batch=8, # number of batch, limit this value based on number of threads you have
        num_utt = 1000, # number of specimen utternance features
        featureFileName='someSpecimenNormFeature_h5_file_path',
        gmmFile='somegmm_h5_file_path',
        minDivCond = False,
        ):
    if os.path.isfile(featureFileName) == False:
        raise ValueError(f"There is no file called {featureFileName} to access specimen features.\n")
    if os.path.isfile(gmmFile) == False:
        raise ValueError(f"There is no file called {gmmFile} to access  gmm trained model.\n")
    if n_col > num_utt:
        raise ValueError(f"There are not enough number of utternances {num_utt} to train TV matrix which has {n_col} number of columns.\n")
    TVMat = initializeSettings(gmm_file=gmmFile, n_col = n_col, divCodn = minDivCond)
    g = tables.open_file(featureFileName,mode='r')
    max_utt = g.root.numberOfUtternance[:][0]
    shmTV = list()
    arTV = list()
    intermediateFileName = gmmFILE[:-3] + "_intermediate_TVResults.h5"
    #creating shared memory to get TV matrix for increment update
    for i in range(no_batch):
        sName, shmAr = sharedMemoryMatrix()
        shmTV.append(sName)
        arTV.append(shmAr)
    print("Shared memory for TV matrix update is created.\n\n")
    stTime = time.perf_counter()
    for l in range(no_iter):
        OldTV = TVMat
        queue = Queue()
        processors = [
                Process(target=TVWorkers,
                    args=(queue,g,num_utt,max_utt,inc_iter,OldTV,shmTV[b])) for b in range(no_batch)
                ]
        for p in processors:
            p.start()
        ASUM = sum([queue.get() for i in range(no_batch)])
        assert(ASUM == no_batch)
        for p in processors: # meeting all threads to a common point in this main function
            p.join()
        con = 0.2 / no_batch # this special number indicate the typical percentage of total data loaded in single iteration
        TVMat = 0.8 * OldTV
        for b in range(no_batch):
            TVMat = TVMat + ( con * arTV[b])
        n_convg = numpy.sum(numpy.sum(numpy.abs((TVMat-OldTV),dtype='float64')))
        OldTV = TVMat
        if n_convg < 0.001:
            break
        if (l % 10) == 0:
            stTime = time.perf_counter() - stTime
            print(f"At {l} iteration, convergence value: {n_convg} and tooks {stTime / 60} minutes.\n")
            print(f"We are saving the intermediate TVMat in a file: {intermediateFileName}.\n")
            svIntermediateTV = tables.open_file(intermediateFileName,'w')
            svIntermediateTV.create_array(svIntermediateTV.root, 'TVMat', TV)
            svIntermediateTV.create_array(svIntermediateTV.root,'itNumber', numpy.array([l+1],dtype='int32'))
            svIntermediateTV.close()

            stTime = time.perf_counter()
    if  n_convg >  0.001:
        print(f"Warning: In bootStrap TV matrix update, parameters are not converged, current convergence value: {n_convg}, consider to do more iteration.\n")
    g.close()
    for b in range(no_batch):
        shmTV[b].close()
        shmTV[b].unlink()
    return TVMat

#-----------------------------------------------------------------------------------------------

def genTVMatrixIncr(
        sp_Feature_File = "someH5SpecimeanNormalizedFeatureFiles", 
        GMM_FILE_NAME="SomeH5GMMFileName",
        MINDIVCOND=False,
        SV_TVMatrix_file = "SomeH5FileSavingTVMat",
        ):

    intermediateFileName = GMM_FILE_NAME[:-3] + "_intermediate_TVResults.h5"
    print(f"Make sure that you dont have a file name (path) {intermediateFileName}, regularly the intermedate TV matrix will be stored in the mentioned file path.\n")
    TV = btStrapTVMatrix(n_col = 150,convg=0.001, inc_iter = 100, no_iter = 150, no_batch=8, 
            num_utt = 10000, featureFileName= sp_Feature_File, gmmFile= GMM_FILE_NAME, minDivCond = MINDIVCOND)
    print(f"Succesfully trainined TV Matrix from features file {sp_Feature_File} and super vector file name {GMM_FILE_NAME}, 
            will store results in {SV_TVMatrix_file}.\n")
    f = tables.open_file(SV_TVMatrix_file,'w')
    f.create_array(f.root, 'TVMat', TV)
    f.close()
    return True

