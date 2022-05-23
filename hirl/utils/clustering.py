import faiss
import torch
import torch.nn as nn
import numpy as np

from sklearn import metrics
from munkres import Munkres

def run_hkmeans(x, num_clusters,  base_temperature=0.2, local_rank=0, niters=20, nredos=5):
    """
    This function is a hierarchical 
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step.
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    
    for seed, num_cluster in enumerate(num_clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = niters
        clus.nredo = nredos
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = local_rank  
        index = faiss.GpuIndexFlatL2(res, d, cfg)  
        if seed==0: # the first hierarchy from instance directly
            clus.train(x, index)   
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
        
        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster 
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

       # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        if seed>0: # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster) # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster))==1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = base_temperature*density/density.mean() 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0: #maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())


        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()    
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def calculate_cost_matrix(C, n_clusters):
    """
    this actually assumes n_clusters_pred == n_clusters_gt
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster j
        for i in range(n_clusters):
            t = C[i, j]
            # cost is defined as number of examples in cluster i - confusion matrix (i, j) 
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset (number of groundtruth clusters)
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order

    Note:
    1. the cost matrix is in size [n_clusters, n_clusters].
    2. The number of cluster in cluster_assignments should be >= n_clusters.
    """
    ## Note: confusion matrix would be of size [M, M], M = max(y_true.max(), cluster_assignments.max())
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None) 
    # confision_matrix[i, j]: number of samples assigned with ground truth label i and clustering label j.
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix) # this is used to find the best 1 to 1 matching.
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def eval_pred(label, pred, calc_acc=False):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    if not calc_acc:
        return nmi, ari, f, -1
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc