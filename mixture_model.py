import numpy
import tables
import psutil
import os
from sklearn.mixture import GaussianMixture

#I am going to introduce two local file variable will indicate percentage of RAM should be used to load data to train gmm
SAFE_MEM_percent = 0.3
CRITICAL_MEM_percent = 0.5
def compute_file_open_mode(file_path):
    mem_available = psutil.virtual_memory().available
    required = os.stat(file_path).st_size
    mode = 0
    if required < int(SAFE_MEM_percent * mem_available):
        mode = 1
        print(f"Current available RAM size is good enough to load file {file_path}.\n")
        print(f"Available memory (in byte) {mem_available} \t required memory to load data {required}.\n")
    elif required < int(CRITICAL_MEM_percent * mem_available):
        mode = 2
        print(f"Warning: Current available RAM size is okay to load the file {file_path}.\n")
        print(f"But it might run slow, remaining compuational space is less, might require more to calculate GMM.\n")
        print(f"Available memory (in byte) {mem_available} \t required memory to load data {required}.\n")
    else:
        mode = 3
        print(f"Current available RAM size is not okay to load the file, we will train the model with less amount of data.\n")
        print(f"Available memory (in byte) {mem_available} \t required memory to load data {required}.\n")
        print(f"If you dont want to train with less data, please  terminate the program now and create some more space.\n")
        print("Or you could continue to train, and use MAP method to further update parameters by SG method.\n")
    return mode
#------------------------------------------------------------------------------------------------------------------------------

def create_andtrain_GMMmodel(features,cluster_comp=1,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7):
    gmModel = GaussianMixture(n_components=cluster_comp,covariance_type=covar_type,tol=convg,max_iter=no_iter,n_init=n_init,random_state=None)
    gmModel.fit(features)
    return gmModel

#tested
def get_GMM_parameters(gModel:GaussianMixture):
    return gModel.n_components, gModel.weights_, gModel.means_, gModel.covariances_
#tested
def get_proba(features,no_compn,model_means,model_covar):
    """given the features, no of components, and the model means and covariance (gaussain multivariate distribution)
    will find the probability of features vector drawn from the given parameters
    """
    if no_compn < 1:
        raise ValueError(f"The number of components in GMM model {no_compn} is less than 1, can't find anything here\n")
    prob = numpy.zeros((features.shape[0],no_compn)) # for each components, we condsider the independent normal multivariate distribution
    k = features.shape[1]
    C = numpy.power(2 * numpy.pi, k / 2)
    for i in range(no_compn):
        vec = model_covar[i,:]
        #vec[numpy.where(vec == 0.0)] = numpy.finfo('float64').eps
        ivec = 1 / vec #for the diagonal matrix, its only to keep inverse of diagonal elements
        denom = C * numpy.sqrt( numpy.prod(vec) , dtype = 'float64')
        feat = features - model_means[i,:]
        var = numpy.sum(feat * ivec * feat, axis=1, dtype='float64') / 2
        prob[:,i] = numpy.exp(-1 * var, dtype='float64') / denom
    prob[numpy.where(prob > 1.0)] = 1.0
    prob[numpy.where(prob < 0.0)] = 0.0
    return prob # this will return for the each feature vector, for each gmm component, it will compute the probability of that feature vector with corresponding parameter
#tested
def get_weighted_prob(prob, weights):
    w_prob = weights * prob 
    Norm = w_prob.sum(axis = 1,dtype='float64')
    w_prob[numpy.where(Norm == 0),:] = weights
    Norm[numpy.where(Norm == 0)] = 1
    return (w_prob.T / Norm).T
#tested
def get_log_likelihood(features,no_compn, model_weights, model_means, model_covar):
    prob_feat = get_proba(features,no_compn,model_means,model_covar)
    arr = numpy.matmul(model_weights,prob_feat.T, dtype='float64') #this is the vector of probability of a feature vector came from the expressed gmm mixture model
    log_prob = numpy.log(arr) # log of the probability
    return log_prob
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""If you want to use the following function, consider to use the pretrained model with good number of features and then update using the following scheme
   don't use the pretrained model trained with less number of features
"""
"""Here we only consider the diagonal covaraince, i.e the different components are independent random variables
"""
#tested
def MAP_update_GMM_parameters(features,no_compn,model_weights,model_means,model_covar,r=16, convg=0.001, n_iter = 300):
    #this relevance factor r in the range of 8 - 20 and it is insensitive to experimental results
    #according to the paper 'Speaker Verification Using Adapted Gaussian Mixture Models'
    if features.shape[0] == 0:
        raise ValueError("There is no data to do any update\n")
    if r < 0:
        raise ValueError(f"r value should be positive but given {r}.\n")
    T = features.shape[0]
    s_iter = 0
    e_convg = 1.0
    SQ_features = numpy.square(features)
    new_means = numpy.zeros((no_compn,features.shape[1]),dtype='float64')
    new_covar = numpy.zeros((no_compn,features.shape[1]),dtype='float64')
    while ( (s_iter <= n_iter) or (e_convg < convg) ):
        prob_feat = get_proba(features,no_compn,model_means,model_covar)
        prob_feat = get_weighted_prob(prob_feat, model_weights)
        stats = prob_feat.sum(axis=0)
        # computing the alpha (a multiplication factor to update the gaussian parameters)
        alpha = stats / (r + stats)
        """
        updating the weights of gmm paramters, this update based on the paper
        -- Speaker Verification using adapted Gaussian mixture models by Douglas A.Reynolds, Thomas F.quatieri and Robert B.Dunn --
        """
        new_weights = ((alpha * stats) / T) + (1 - alpha) * model_weights
        new_weights /= new_weights.sum()
        #assert(new_weights.sum()==1.0) # there is a first sign of lossing numerical precision here
        """
        --simillarly updating the model mean and also covariance parameter (if it is diagonal)
        """
        for i in range(no_compn):
            if numpy.abs(stats[i]) > numpy.finfo(float).eps:
                Em = (numpy.matmul(prob_feat[:,i],features)) / stats[i] # computing weighted average of features with respect to i-th component
                Esq = (numpy.matmul(prob_feat[:,i],SQ_features)) / stats[i] # computing weighted average of squared features with respect to i-th component
                new_means[i,:] = alpha[i] * Em + (1 - alpha[i]) * model_means[i,:]
                new_covar[i,:] = alpha[i] * Esq + (1 - alpha[i]) * (model_covar[i,:] + numpy.square(model_means[i,:])) - numpy.square(new_means[i,:])
            else:
                new_means[i,:] =  (1 - alpha[i]) * model_means[i,:]
                new_covar[i,:] =  (1 - alpha[i]) * (model_covar[i,:] + numpy.square(model_means[i,:])) - numpy.square(new_means[i,:])
        s_iter += 1
        e_convg = numpy.sum(numpy.abs(new_means - model_means), dtype='float64')
        model_weights = new_weights
        model_means = new_means
        model_covar = new_covar
    #we are updating the means, variance, weights by convergence repeated manner MAP (Maximum A posterior probability method)
    return new_weights, new_means, new_covar
#tested
def MAP_update_GMM_mean_only(features,no_compn,model_weights,model_means,model_covar,r=16, convg=0.001, n_iter = 300):
    #this relevance factor r in range of 8 - 20 and it is insensitive to experimental results
    #according to the paper 'Speaker Verification Using Adapted Gaussian Mixture Models'
    """In this function, we only update the mean of the gmm mixture model, but we still need the 
    model_covar to calculate the model covariance, the basic assumption is we trained the initial mixture model with large data sets
    the specific person trains dont have much difference in his square of mean variation but 
    he is shifted in the first order terms in the sense there is difference in the energy spectrum or the region of frequency 
    """
    if features.shape[0] == 0:
        raise ValueError("There is no data to do any update\n")
    if r < 0:
        raise ValueError(f"r value should be positive but given {r}\n")
    T = features.shape[0]
    e_convg = 1
    s_iter = 0
    new_means = numpy.zeros((no_compn,features.shape[1]), dtype='float64')
    while ( (s_iter <= n_iter) or (e_convg < convg) ):
        prob_feat = get_proba(features,no_compn,model_means,model_covar)
        prob_feat = get_weighted_prob(prob_feat, model_weights)
        stats = prob_feat.sum(axis=0)
        # computing the alpha (a multiplication factor to update the gaussian parameters)
        alpha = stats / (r + stats)
        for i in range(no_compn):
            if numpy.abs(stats[i]) > numpy.finfo(float).eps:
                Em = (numpy.matmul(prob_feat[:,i],features)) / stats[i] # computing weighted average of features with respect to i-th component
                new_means[i,:] = alpha[i] * Em + (1 - alpha[i]) * model_means[i,:]
            else:
                new_means[i,:] =  (1 - alpha[i]) * model_means[i,:]
        e_convg = numpy.sum(numpy.abs(new_means - model_means))
        s_iter += 1
        #model_weights = new_weights
        model_means = new_means
    # now we are using iterative, convergence i,e MAP to update parameters for specific features
    return model_means
#--------------------------------------------------------------------------------------------------------------------------------------
"""Now, we will have function to generate UBM of voiced data
"""
#tested
#if you have bigger corpus, run this command in network instead of your local machine
# with the combination of gmm model and MAP, use the stochastically gradient decent method to update the parameters
def generate_ubm(file_path = 'Some_Norm_features_hfile', res_path = 'gmm_ubm_parameters', N_component = 1024, init_mean = None):
    if os.path.isfile(file_path) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    f = tables.open_file(file_path,mode = 'r')
    mode = compute_file_open_mode(file_path)
    if (mode == 1) or (mode == 2):
        features = f.root.Norm_features[:,:]
    else:
        suitable_size = int( SAFE_MEM_percent * mem_available) 
        features = f.root.Norm_features[0:suitable_size,:]
    f.close()
    if init_mean is None:
        UBM_model = create_andtrain_GMMmodel(features,cluster_comp=N_component,convg=0.0001,no_iter=400,n_init=7, init_mean = init_mean)
    else:
        UBM_model = create_andtrain_GMMmodel(features,cluster_comp=N_component,convg=0.0001,no_iter=400,n_init=1, init_mean = init_mean)

    print(f"Succesfully trainined UBM (diagonal covar) model from the features file {file_path}, will store results in {res_path}.\n")
    _, weights, means, covar = get_GMM_parameters(UBM_model)
    f = tables.open_file(res_path,'w')
    f.create_array(f.root, 'info_dim', numpy.array([N_component, features.shape[1]]))
    f.create_array(f.root, 'weights', weights)
    f.create_array(f.root, 'means', means)
    f.create_array(f.root, 'covar', covar)
    f.close()
    return N_component,weights,means,covar
#----------------------------------------------------------------------------------------------------------------------------------------------
