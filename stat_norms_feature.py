import numpy
import os
import tables
import errno
SEG_SIZE = 100000

#In order to do proper pre normalization with good statisics scale, use this function
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

