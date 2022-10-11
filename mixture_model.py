import numpy
from sklearn.mixture import GaussianMixture

def create_andtrain_GMMmodel(features,cluster_comp=1,covar_type = 'diag',convg=0.001,no_iter=300,n_init=7):
    gmModel = GaussianMixture(n_components=cluster_comp,covariance_type=covar_type,tol=convg,max_iter=no_iter,n_init=n_init,random_state=None)
    gmModel.fit(features)
    return gmModel

def get_GMM_parameters(gModel:GaussianMixture): 
    return gModel.n_components, gModel.weights_, gModel.means_, gModel.covariances_

def get_proba(features,no_compn,model_means,model_covar):
    if no_compn < 1:
        raise ValueError(f"The number of components in GMM model {no_compn} is less than 1, can't find anything here\n")
    prob = numpy.zeros((features.shape[0],no_compn))
    k = features.shape[1]
    C = numpy.power(2 * numpy.pi, k / 2)
    #There is some problem in the code, when the number of datas is too less or the covariance is near to zeros, then the probability calculation goes crazy
    #For sure the following code should be computed by mutli thread way to reduce real time computations
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
    return prob

def get_weighted_prob(prob, weights):
    w_prob = weights * prob 
    Norm = w_prob.sum(axis = 1,dtype='float64')
    w_prob[numpy.where(Norm == 0),:] = weights
    Norm[numpy.where(Norm == 0)] = 1
    return (w_prob.T / Norm).T

def get_log_likelihood(features,no_compn,model_weights,model_means,model_covar):
    prob_feat = get_proba(features,no_compn,model_means,model_covar)
    arr = numpy.matmul(model_weights,prob_feat.T, dtype='float64')
    log_prob = numpy.log(arr)
    return log_prob

# as of now we are succesful only on gmm (i.e speaker verification) not on gmm-ubm to do the Identification stuff
#in order to enter this world, we have to go forward and make the following code works. May be use the following code if you have really 
#trained the gmm with huge data sets

def MAP_update_GMM_parameters(features,no_compn,model_weights,model_means,model_covar,r=16):
    #this relevance factor r in the range of 8 - 20 is insensitive to experimental results
    #according to the paper Speaker Verification Using Adapted Gaussian Mixture Models
    if features.shape[0] == 0:
        raise ValueError("There is no data to do any update\n")
    T = features.shape[0]
    prob_feat = get_proba(features,no_compn,model_means,model_covar)
    prob_feat = get_weighted_prob(prob_feat, model_weights)
    stats = prob_feat.sum(axis=0)
    # computing the alpha (a multiplication factor to update the gaussian parameters)
    if r == 0:
        alpha = numpy.ones((no_compn,))
    else:
        alpha = stats / (r + stats)
    """
    updating the weights of gmm paramters, this update based on the paper
    -- Speaker Verification using adapted Gaussian mixture models by Douglas A.Reynolds, Thomas F.quatieri and Robert B.Dunn --
    """
    new_weights = ((alpha * stats) / T) + (1 - alpha) * model_weights
    new_weights /= new_weights.sum()
    assert(new_weights.sum()==1.0)
    """
    --simillarly updating the model mean and also covariance parameter (if it is diagonal)
    """
    SQ_features = numpy.square(features)
    new_means = numpy.zeros((no_compn,features.shape[1]),dtype='float64')
    new_covar = numpy.zeros((no_compn,features.shape[1]),dtype='float64')
    for i in range(no_compn):
        if numpy.abs(stats[i]) > numpy.finfo(float).eps:
            Em = (numpy.matmul(prob_feat[:,i],features)) / stats[i] # computing weighted average of features with respect to i-th component
            Esq = (numpy.matmul(prob_feat[:,i],SQ_features)) / stats[i] # computing weighted average of squared features with respect to i-th component
            new_means[i,:] = alpha[i] * Em + (1 - alpha[i]) * model_means[i,:]
            new_covar[i,:] = alpha[i] * Esq + (1 - alpha[i]) * (model_covar[i,:] - numpy.square(model_means[i,:])) - numpy.square(new_means[i,:])
        else:
            new_means[i,:] =  (1 - alpha[i]) * model_means[i,:]
            new_covar[i,:] =  (1 - alpha[i]) * (model_covar[i,:] - numpy.square(model_means[i,:])) - numpy.square(new_means[i,:])
    return new_weights, new_means, new_covar

