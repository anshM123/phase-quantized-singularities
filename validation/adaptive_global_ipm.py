"""Front-adapted nonuniform radial IPM solver with separate sparse-Newton formulation.
A numerical experiment; not a rigorous existence or convergence certificate.
"""
import sys,gc,time,json,traceback,math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ncs-submission/code/lib"))
from ipm_global import IPMGlobal
LAM=1/7.346
SRC=str(ROOT/"ncs-submission/data/profiles/ipm_state_z7.3460_Nb32_hs0.00625.npy")
def adaptive_grid(smin=-10.,smax=14.,hmin=.00625,hfar=.15,sigma=.7):
 s=[smin]
 while s[-1]<smax-1e-10:
  h=hmin+(hfar-hmin)*(1-np.exp(-.5*(s[-1]/sigma)**2))
  s.append(min(smax,s[-1]+h))
  if len(s)>20000:raise RuntimeError("mesh too large")
 if s[-1]-s[-2]<.05:s[-2:]=[smax]
 return np.array(s)
def fd(s,der,width,shift=0):
 n=len(s);rr=[];cc=[];vv=[];w=width//2
 for i in range(n):
  lo=max(0,min(i-w-shift,n-width));ids=np.arange(lo,lo+width)
  d=s[ids]-s[i];scale=max(abs(d));x=d/scale
  V=np.vander(x,N=width,increasing=True).T
  rhs=np.zeros(width);rhs[der]=math.factorial(der)/scale**der
  co=np.linalg.solve(V,rhs)
  rr.extend([i]*width);cc.extend(ids);vv.extend(co)
 return sp.csr_matrix((vv,(rr,cc)),shape=(n,n))
class Adaptive(IPMGlobal):
 def __init__(self,smin=-10,smax=14,hmin=.00625,hfar=.15,sigma=.7,Nb=32):
  super().__init__(s_min=smin,s_max=smax,hs=hfar,Nb=Nb)
  self.s=adaptive_grid(smin,smax,hmin,hfar,sigma);self.Ns=len(self.s)
  I=sp.identity(self.n,format="csr");Is=sp.identity(self.Ns,format="csr")
  self.Ds=sp.kron(fd(self.s,1,9),I,format="csr")
  self.Du=sp.kron(fd(self.s,1,9,1),I,format="csr")
  self.Dss=sp.kron(fd(self.s,2,11),I,format="csr")
  self.Dbeta=sp.kron(Is,sp.csr_matrix(self.Db),format="csr")
  self.Dbeta2=sp.kron(Is,sp.csr_matrix(self.Dbb),format="csr")
  self.N=self.Ns*self.n
  S,Bt=np.meshgrid(self.s,self.beta,indexing="ij")
  self.S=S.ravel();self.CB=np.cos(Bt).ravel();self.SB=np.sin(Bt).ravel()
  self.CB[np.isclose(Bt.ravel(),np.pi/2)]=0.
  self.eS=np.exp(-self.S)
  j=np.arange(self.N)//self.n;k=np.arange(self.N)%self.n
  self.inflow=j==0;self.xbc=(k==0)|(k==self.n-1)
  self.xfar=(j==self.Ns-1)&(~self.xbc);self.xreg=(j==0)&(~self.xbc)
  self.k0_smin=0
def solve(hmin,nb):
 t0=time.time();G=Adaptive(hmin=hmin,Nb=nb)
 U=G.initial_from_march(SRC,LAM,Nb_src=32,hs_src=.00625,s_start=-20)
 hist=[]
 for it in range(17):
  res=G.residual(U);err=float(np.max(abs(res)))
  D=1+U[-1]+G.Dbeta@U[2*G.N:3*G.N]
  row=dict(iter=it,res=err,lambda_=float(U[-1]),z=float(1/U[-1]),min_D=float(np.min(D)))
  hist.append(row);print("ITER",json.dumps(row),flush=True)
  if err<1e-9 and float(np.min(D))>0:break
  delta=splu(G.jacobian(U)).solve(-res)
  got=False
  for j in range(18):
   fac=2.**(-j);V=U+fac*delta
   if abs(V[-1]-LAM)>.007:continue
   D2=1+V[-1]+G.Dbeta@V[2*G.N:3*G.N]
   if np.min(D2)<=0:continue
   new=float(np.max(abs(G.residual(V))))
   if new<err*(1-1e-4*fac):U=V;got=True;break
  if not got:
   row["failure"]="line search or branch constraint"
   break
 final=float(np.max(abs(G.residual(U))))
 return dict(hmin=hmin,Nb=nb,Ns=G.Ns,min_s_step=float(np.min(np.diff(G.s))),
    computed_lambda=float(U[-1]),computed_z=float(1/U[-1]),
    initial_residual=hist[0]["res"],final_residual=final,
    converged=bool(final<1e-9),history=hist,seconds=round(time.time()-t0,1))
if __name__=="__main__":
 s=adaptive_grid();f=s**5
 print("POLYNOMIAL_TEST",float(np.max(abs(fd(s,1,9)@f-5*s**4))),float(np.max(abs(fd(s,2,11)@f-20*s**3))),flush=True)
 cases=[(.025,32),(.0125,32),(.00625,32),(.00625,40)]
 out=[]
 for h,n in cases:
  print("CASE",h,n,flush=True)
  try:r=solve(h,n)
  except Exception as e:r=dict(hmin=h,Nb=n,error=str(e),traceback=traceback.format_exc())
  out.append(r);print("ADAPT_RESULT",json.dumps({k:v for k,v in r.items() if k!="history"}),flush=True)
  (ROOT/"validation/adaptive_results.json").write_text(json.dumps(out,indent=2));gc.collect()
 print("DONE",flush=True)
