import math
import numpy as np                      # numpy for basic linear algebra
from itertools import combinations      # combinations for enumerating facets

def binomial(x, y):
    try:
        binom = math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        binom = 0
    return binom

class KAPCycle:
    def __init__(self,N):
        self.N = N

    def ap_bound(self):
        N = self.N
        return N * binomial(N-1,(N-1)//2)

    def lifted_supp(self):
        N = self.N
        A = np.zeros((2*N+1,N),dtype=int)
        for i in range(N):
            A[2*i  ,N-1] = 1                       # lifting of 1
            A[2*i+1,N-1] = 1                       # lifting of 1
            if i < N-2:
                A[2*i  ,i+1] = -1
                A[2*i+1,i+1] =  1
            if i < N-1:
                A[2*i  ,i] =  1
                A[2*i+1,i] = -1
            else:
                A[2*i  ,0] = -1
                A[2*i+1,0] =  1
                if N % 2 == 0:                      # for the even N cases
                    A[2*i  ,N-1] = 2                # we have to use lifting of 2
                    A[2*i+1,N-1] = 2
        return A

    def __facet_indices(self):
        ls = []
        N = self.N
        if N % 2 == 0:                              # even N case
            r = N // 2                              # number of negative entries
            for ns in combinations(range(N),r):
                idx = np.ones((self.N),dtype=int)
                for k in ns:
                    idx[k] = -1
                ls.append (idx)
        else:                                       # odd N case
            r = N // 2                              # number of negative entries
            for j in range(N):                      # for each j
                ds = list(range(N))
                ds.remove(j)
                for ns in combinations(ds,r):
                    idx = np.ones((self.N),dtype=int)
                    idx[j] = 0
                    for k in ns:
                        idx[k] = -1
                    ls.append (idx)
        return ls

    def T_matrix(self,idx,k):
        N = self.N
        T = np.zeros((N+1,N))
        T[0,0] = -idx[0]
        T[0,N-1] = 2
        for i in range(1,N):
            T[i,i-1] =  idx[i]
            T[i,i  ] = -idx[i]
            T[i,N-1] = 1
        mask = np.ones((N+1),dtype=bool)
        mask[k] = False
        return T[mask,...]

    def T_matrix_reduced(self,idx,k):
        N = self.N
        n = N - 1
        if N % 2 == 0:                      # for even N
            T = np.zeros((N,n),dtype=int)
            T[0,0] = -idx[0]
            for i in range(1,N):
                T[i,i-1] =  idx[i]
                if i < N-1:
                    T[i,i] = -idx[i]
            mask = np.ones((N),dtype=bool)
            mask[k] = False
            return T[mask,...]
        else:
            T = np.zeros ((n,n), dtype=int)
            r = 0
            for k in range(N):              # for each facet index position
                if idx[k] != 0:
                    assert r < N
                    if k < n:
                        T[r,k] = -idx[k]
                    if k > 0:
                        T[r,k-1] = idx[k]
                    r = r + 1
            assert r == n
            return T

    def __inormal(self,idx,k):
        N = self.N
        x = np.empty((N,1),dtype=int)
        x[N-1] = 1
        if N % 2 == 0:                      # for even N
            if idx[0] < 0:                  # if we have negative leading
                x = -self.__inormal(-idx,k) # double flip this
                x[N-1] = 1                  # still need to be upward pointing though
            else:
                x[0] = idx[0]
                for i in range(1,N-1):
                    x[i] = x[i-1] + idx[i]
                    # for j in range(i+1):
                    #     x[i] += idx[j]
                for j in range(k):
                    x[j] += 1
        else:                               # for odd N
            x[0] = idx[0]
            for i in range(1,N-1):
                x[i] = x[i-1] + idx[i]
        return x

    def cells(self):
        N = self.N
        # A = self.lifted_supp()
        cs = []
        for idx in self.__facet_indices():
            if N % 2 == 0:
                for k in range(N):
                    if idx[k] == idx[0]:
                        # T = self.T_matrix(idx,k)
                        R = self.T_matrix_reduced(idx,k)
                        x = self.__inormal(idx,k)
                        # h = np.dot(A,x)
                        C = {
                            # "index"   : idx,
                            "origmat" : R,
                            "inormal" : x
                            # "heights" : h,
                            # "islower" : 0 == (h < 0).sum(),
                            # "lowpts"  : (h == 0).sum()
                        }
                        cs.append(C)
            else:
                R = self.T_matrix_reduced(idx,0)
                x = self.__inormal(idx,0)
                # h = np.dot(A,x)
                C = {
                    # "liftmat" : T,
                    # "index"   : idx,
                    "origmat" : R,
                    "inormal" : x
                    # "heights" : h,
                    # "islower" : 0 == (h < 0).sum(),
                    # "lowpts"  : (h == 0).sum()
                }
                cs.append(C)

        return cs
