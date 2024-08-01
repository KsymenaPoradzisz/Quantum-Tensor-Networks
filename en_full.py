#Author: Piotr Czarnik

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh


Z = np.diag([1,-1])
X = np.array([[0., 1.],[1., 0.]])

# generates a random quntum state for n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def random(n):
  x = np.random.rand(2**n)+1.j*np.random.rand(2**n)
  x = x/np.sqrt(np.vdot(x,x))
  x = np.reshape(x,list(np.repeat(2,n)))
  return x

# generates a random ground state of a classical 1D Ising model with n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def Is_GS(n):
  x = np.random.rand(2)+1.j*np.random.rand(2)
  x = x/np.sqrt(np.vdot(x,x))
  v = np.zeros(2**n,dtype=np.complex128)
  v[0] = x[0]
  v[2**n-1] = x[1]
  v = np.reshape(v,list(np.repeat(2,n)))
  return v

# apply one-dimensional quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
#H = -\sum_{i=0}^{n-2} Z_i Z_j - h \sum_{i=0}^{n-1} X_i - h_Z \sum_{i=0}^{n-1} Z_i.
# to a state given by  a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def apply_H(psi,n,h,hz):
  sv = np.shape(psi)
  #print(sv)
  psio = np.zeros(sv)
  #print(np.shape(psio))
  op2 = np.transpose(np.tensordot(-Z,Z,axes=((),())),(0,2,1,3))
  op = -h*X-hz*Z
  for i in range(n-1):
    psit = np.tensordot(op2,psi,axes=((2,3),(i,i+1)))
    psit = np.transpose(np.reshape(psit,(4,2**(i),2**(n-i-2))),(1,0,2))
    psit = np.reshape(psit,sv) 
    psio = psio+psit
  for i in range(n):
    psit = np.tensordot(op,psi,axes=((1,),(i,)))    
    psit = np.transpose(np.reshape(psit,(2,2**(i),2**(n-i-1))),(1,0,2)) 
    psit = np.reshape(psit,sv) 
    psio = psio+psit
  return psio

# the same as apply_H but psi_{0,1,2,\dots,n-1} reshaped into a vector (required for the ground state computation)
def apply_H_wrap(psi,n,h,hz):
  sv = [2 for i in range(n)]
  #print(sv)
  psir = np.reshape(psi,sv)
  psio = apply_H(psir,n,h,hz)
  return np.reshape(psio,-1)
  

# compute energy of  the quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
def en_H(psi,n,h, hz):
  psio = apply_H(psi,n,h,hz)
  EV = np.vdot(np.reshape(psi,-1),np.reshape(psio,-1))
  return np.real(EV)

# compute the ground state energy of  the quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
def en_GS(n,h,hz):
  apply_H_hand = lambda x : apply_H_wrap(x,n,h,hz)
  dmx = 2**n
  A = LinearOperator((dmx,dmx), matvec=apply_H_hand)
  evals_small, evecs_small = eigsh(A, 1, which='SA')
  return evals_small[0]


n=7 # the qubit number
h=3 # the transverse field
hz = 0.01 # the longitudinal fiels
psi = random(n) # the random state
en = en_H(psi,n,h,hz)
enGS = en_GS(n,h,hz)

print("energy of a random state is "+ str(en))
print("the ground state energy is " +str(enGS))

