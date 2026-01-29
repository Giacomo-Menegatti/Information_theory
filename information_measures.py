import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_unique_probs(x):
    '''This function digitizes values and returns the probability of each bin and the index of each value'''
    _, unique_inverse, unique_counts =  np.unique(x, return_inverse=True, return_counts=True)
    # Return the probabilities and the inverse
    return np.array(unique_counts/np.sum(unique_counts)), unique_inverse
    

def binned_h(x, y, bins):
    '''This function computes the entropy of x and the mutual information of x and y'''
    # Bin the values by taking the bin index for each value with digitize and then calling bins[...]
    binned_x = bins[np.digitize(x,bins)]
    p_x, _ = get_unique_probs(binned_x)

    #Compute the entropy of X
    ENTROPY_X = - np.sum(p_x * np.log(p_x))

    # Get the probabilities and the inverse index of Y (i.e., the indexes of X GIVEN Y)
    p_y, inverse_index_y = get_unique_probs(y)
    # Compute the conditional entropy X|Y
    CONDITIONAL_X_Y = 0

    for j in range(len(p_y)):   # Cycle over all indexes of y
        # Compute p_x|y by taking the binned values corresponding to the j-th value
        p_x_given_y, _ = get_unique_probs(binned_x[inverse_index_y == j])
        # Compute the conditional entropy for y_j
        CONDITIONAL_X_Y += -p_y[j] * np.sum(p_x_given_y * np.log(p_x_given_y))

    #print(f'The degenerate limit is {np.log(len(x))}')
    return ENTROPY_X, ENTROPY_X - CONDITIONAL_X_Y

def kolchinsky_h(x, y, sigma, mode='upper'):
    ''' Computes the entropy upper bound using a gaussian KDE'''
    P = len(x)  #Number of samples
    c = 1 if mode=='upper' else 4
    # compute the distance matrix, where d_ij = |x_i - x_j|**2
    # pdist computes the pairwise distance between each pair of vectors, squreform rearranges the output in the distance matrix
    distances = pdist(x, metric='sqeuclidean')
    D = squareform(distances)
    #print(f'Distance std is {np.std(distances)}')
    # Matrix of gaussian distances
    G = np.exp(-D/(2*c*sigma**2))

    ENTROPY_X = -1.0/P * np.sum(np.log(1.0/P * np.sum(G, axis=1)))

    # Get the conditional entropy
    CONDITIONAL_X_Y = 0.0
    _, inverse_index_y, counts_y = np.unique(y, return_inverse=True, return_counts=True)

    for j, P_j in enumerate(counts_y):
        # Keep only the entries corresponding to the j-th value
        G_given_y = G[inverse_index_y == j][:, inverse_index_y == j]
        # Compute the conditional entropy for the j-th value
        CONDITIONAL_X_Y += - 1.0/P * np.sum( np.log(1.0/P_j * np.sum(G_given_y, axis=1)))

    return ENTROPY_X, ENTROPY_X - CONDITIONAL_X_Y