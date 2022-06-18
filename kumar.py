import networkx as nx
import numpy as np
from community import community_louvain

def weight_matrix(W:list): 
    # gets weight list and returns hyperedges weight matrix 
    m = len(W)
    diagonal = [w*1.0 for w in W]
    W = np.zeros(shape=(m,m))
    np.fill_diagonal(W,diagonal)
    return W

def cal_De(H): 
    # returns edge degree matrix 
    HT = np.transpose(H)
    m = np.shape(H)[1] # m is number of hyperedges
    De = np.zeros(shape=(m,m))
    diagonal = [sum(item) for item in HT]
    np.fill_diagonal(De,diagonal)
    return De

def cal_Dv(H,W): 
    # returns vertex degree matrix
    n = np.shape(H)[0] # n is number of vertices
    Dv = np.zeros(shape=(n,n))
    W = W.diagonal()
    diagonal = [np.dot(vertex,W) for vertex in H]
    np.fill_diagonal(Dv,diagonal)
    return Dv


def irmm(H, W=None , threshold=0.01 , verbose=False, alpha = 0.5):
    # Iteratively Reweighted Modularity Maximization (IRMM) Algorithm for hypergraph clustering, kumar et al. 2020
    m = np.shape(H)[1] # number of hyperedges
    if W==None:
        W = [1]*m  # Default hyperedges weight are [1,..,1] 
    else:
        if len(W) != m:
            return "Error: Weight W is not compatible with Hypergraph H!"
    W = weight_matrix(W)
    Dv = cal_Dv(H,W) # vertex degree matrix
    De = cal_De(H) # edge degree matrix 

    iteration = 0
    while True:
        iteration+=1
        if verbose:print("===========\nIteration",iteration)
        if verbose:print("W:\n",W)
        # compute reduced adjacency matrix (A<-H.W.(De-I)^-1.H^T)
        HW = np.matmul(H,W)
        De_I = De - np.identity(n=np.shape(De)[0])
        De_I_inv = np.linalg.inv(De_I) 
        HWDe_I_inv = np.matmul(HW,De_I_inv)
        HT = np.transpose(H)
        A = np.matmul(HWDe_I_inv,HT)
        np.fill_diagonal(A, 0) # zero out the diagonals of A
        if verbose:print("A:\n",A)

        # return number of clusters and cluster assignments
        G = nx.from_numpy_matrix(A) 
        cluster_assignments = community_louvain.best_partition(graph=G,weight='weight',randomize=False)
        if verbose:print('cluster_assignments:',cluster_assignments)
        cluster_ids = np.unique(list(cluster_assignments.values()))
        c = len(cluster_ids) # number of clusters

        
        # Compute new weight for each hyperedge
        if verbose:print("\nreweighting W:\n---")
        Wـprev = W.copy()  # keeps previous weight matrix
        for ei in range(m):
            e = HT[ei]
            e_nodes = [index for index,value in enumerate(e) if value == 1]
            delta_e = De[ei,ei] # = len(e_nodes)
            if verbose:print('hyperedge',ei,"=",e_nodes,",delta_e=",delta_e)
            # Compute the number of nodes in each cluster
            cluster_nodes = dict()
            k = dict() 
            for i in cluster_ids:
                # Set of nodes in cluster i
                cluster_nodes[i] = [key for key,value in cluster_assignments.items() if value == i] 
                k[i] = len([value for value in cluster_nodes[i] if value in e_nodes]) # k[i] = |e intersection C[i]|
            if verbose:print('cluster_nodes:',cluster_nodes)
            if verbose:print('k for this hyperedge:',k)
            if verbose:print("---")
            # Compute new weight
            w_e = 0  # w'(e)
            for i in cluster_ids:
                w_e += (1/m) * (1/(k[i] + 1)) * (delta_e + c)
            # Take moving average with previous weight
            W[ei,ei] =  alpha*W[ei,ei] + (1-alpha)*w_e
        if verbose:print("W is updated.\n")
        if np.linalg.norm(W - Wـprev) < threshold:
            if verbose:print("==========\nIRMM Done!\n")
            break
    return cluster_assignments


'''
# test input 

H1 = np.array([   # H(i,e): is node i in hyperedge e ? 
    [0,0,1,0],
    [0,0,1,0],
    [0,1,1,0],
    [0,1,1,0],
    [1,1,0,1],
    [0,1,0,0],
    [1,0,0,1],
    [1,0,0,1],
    [0,0,0,1],
    [0,0,0,1]
])
W1 = [1,1.5,1,1.5]
H2 = np.array([   # H(i,e): is there node i in hyperedge e? 
    [1,1,0,0,0],
    [1,1,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,1,1,0,0],
    [0,1,1,0,1],
    [0,0,1,0,1],
    [0,0,0,1,0],
    [0,0,0,1,0],
    [0,0,0,1,0]
])
W2 = [1,2,1,1,1]
H3 = np.array([   # H(i,e): is node i in hyperedge e ? 
    [1,0],
    [1,0],
    [1,1],
    [0,1],
    [0,1],
    [0,1]
]) 
print(irmm(H = H3, W=[1,3]))



H4 = np.array([   # H(i,e): is node i in hyperedge e ? 
    [1,1,0,0],
    [1,0,1,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,1,0,1],
    [0,1,0,1],
    [0,1,0,0]
]) 
print(irmm(H=H4,W=[1,1.2,1,1.5],verbose=True))
'''