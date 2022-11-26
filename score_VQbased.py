import numpy
import tables
import os
import errno
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA as IPCA

def nearsestVQ_specimen_features(VQ_file, sp_file):
    """Given the VQ code book file (.h5 file), grap the VQ cluster, 
        given the speicmen feature file sp_file (.npy file), grap the features and label them (map) index of VQ cluster
    """
    #VQ_file should have clustered VQ codebook from the general voice data sets (CMNV of specific language features)
    if os.path.isfile(VQ_file) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), VQ_file)
    #sp_file should have the features of specific registred or claimed voice features (with same CMNV)
    if os.path.isfile(sp_file) == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sp_file)
    f = tables.open_file(VQ_file,mode="r")
    VQ_size, num_feat = f.root.dim[0], f.root.dim[1]
    codebook_center = f.root.VQ[:,:]
    f.close()
    sp_features = numpy.load(sp_file)
    code_cluster = KMeans( n_clusters=VQ_size, init=codebook_center, n_init=1, max_iter=2)
    code_cluster.fit(codebook_center)
    data_label = code_cluster.predict(sp_features)
    return data_label

def simple_target_test_cosine_measures(sp_feat, sp_feat_label, test_feat, test_feat_label):
    """Given the specimen features and label (sp_feat, sp_feat_label)
        testing with the test features and lable (tst_feat, tst_feat_label)
        will return the (avg) cosine measure of test and specimen features
    """
    score_lst = list()
    for t in range(test_feat_label.shape[0]):
        for s in range(sp_feat_label.shape[0]):
            if (sp_feat_label[s] == test_feature_label[t]):
                sc =  numpy.dot(sp_feat[s], test_feat[t]) / ( numpy.sqrt(numpy.dot(sp_feat[s], sp_feat[s])) * numpy.sqrt(numpy.dot(test_feat[t],test_feat[t])))
                score_lst.append(sc)
    Avg_score = 0
    if (len(score_lst) == 0):
        Avg_score = -5
    else:
        Avg_score = sum(score_lst) / len(score_lst)
    return Avg_score


def dimreduction_byPCA(features,n_components=25,random_state=42,p=None):
    """Given the features (number of features vectors cross number of features),
        reduce its dimenstion by linear methods (incremental principle component analyse)
        return the reduced features
    """
    if p == None:
        p = IPCA(n_components=n_components,batch_size=features.shape[0]).partial_fit(features)
    else:
        p.partial_fit(features)
    #note here, check always that your features matrix mean is 0 along the axis 0 i.e along the number of data samples
    return p, p.components_, p.transform(features)
    
def linear_trasformation(features, trans_matrix):# with pretrainted transformation, transform the features
    """With the trained dimensionality reduction tranformation (trans_matrix)
        transfer the features using this linear transofmration (matrix multiplication)
    """
    return numpy.matmul(trans_matrix, features.T).T

