"""Finer seeded IPM global-mesh experiment; not a rigorous continuum certificate."""
from pathlib import Path
import sys,time,json,gc,traceback
import numpy as np
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ncs-submission/code/lib"))
from ipm_global import IPMGlobal
from bq_global import fd_matrix
from scipy.sparse import kron,eye
ORIGIN=1/7.346
FILE=str(ROOT/"ncs-submission/data/profiles/ipm_state_z7.3460_Nb32_hs0.00625.npy")
CASES=[(.05,40),(.025,32),(.025,40)]
out=[]
def norms(G,U):
 R,Om,X,lam=G.unpack(U)
 Ns=G.Ns;n=G.n
 Ds=kron(fd_matrix(Ns,G.hs,order=8,der=1),eye(n,format="csr"))
 Dss=kron(fd_matrix(Ns,G.hs,order=8,der=2),eye(n,format="csr"))
 Rd=Ds@R;Rb=G.Dbeta@R;Xd=Ds@X
 D=1+lam+G.Dbeta@X;w=-(2*X+Xd)
 t=-lam*R+D*Rd+w*Rb
 p=Dss@X+4*Xd+4*X+G.Dbeta2@X+Om
 sl=(G.s>=-7)&(G.s<=4)
 t=t.reshape(Ns,n)[sl,1:-1];p=p.reshape(Ns,n)[sl,1:-1]
 return dict(transport_max=float(np.max(abs(t))),transport_rms=float(np.sqrt(np.mean(abs(t)**2))),
             poisson_max=float(np.max(abs(p))),poisson_rms=float(np.sqrt(np.mean(abs(p)**2))))
def run(hs,nb):
 start=time.time()
 G=IPMGlobal(s_min=-10,s_max=14,hs=hs,Nb=nb)
 U=G.initial_from_march(FILE,ORIGIN,Nb_src=32,hs_src=.00625,s_start=-20)
 rows=[]
 for it in range(18):
  res=G.residual(U);err=float(np.max(abs(res)));lam=float(U[-1]);D=1+lam+G.Dbeta@U[2*G.N:3*G.N]
  rec=dict(step=it,residual=err,lambda_=lam,z=1/lam,min_D=float(D.min()));rows.append(rec)
  print("ITER",json.dumps(rec),flush=True)
  if err<1e-9 and D.min()>0:break
  delta=splu(G.jacobian(U)).solve(-res)
  accept=False
  for j in range(16):
   t=2.0**(-j);V=U+t*delta
   if abs(V[-1]-ORIGIN)>.007:continue
   Dv=1+V[-1]+G.Dbeta@V[2*G.N:3*G.N]
   if float(np.min(Dv))<=0:continue
   er2=float(np.max(abs(G.residual(V))))
   if er2<err*(1-1e-4*t):
    U=V;accept=True;break
  if not accept:
   rec["failure"]="no branch-preserving decreasing step";break
 final=float(np.max(abs(G.residual(U))))
 D=1+U[-1]+G.Dbeta@U[2*G.N:3*G.N]
 ok=final<1e-9 and float(np.min(D))>0 and abs(U[-1]-ORIGIN)<.007
 row=dict(radial_h=hs,Nb=nb,solved_lambda=float(U[-1]),solved_z=float(1/U[-1]),target_lambda=ORIGIN,initial_residual=rows[0]["residual"],final_residual=final,min_transport=float(np.min(D)),discrete_converged=bool(ok),history=rows,elapsed_s=round(time.time()-start,1))
 if ok:row["cross_discretization"]=norms(G,U)
 np.savez_compressed(ROOT/"validation"/f"ultra_h{hs}_Nb{nb}.npz",U=U,s=G.s,beta=G.beta)
 return row
for hs,nb in CASES:
 print("CASE",hs,nb,flush=True)
 try:row=run(hs,nb)
 except Exception as exc:row=dict(radial_h=hs,Nb=nb,error=str(exc),traceback=traceback.format_exc())
 out.append(row);print("ULTRA_RESULT",json.dumps({k:v for k,v in row.items() if k!="history"}),flush=True)
 (ROOT/"validation/ultra_results.json").write_text(json.dumps(out,indent=2))
 gc.collect()
print("DONE",flush=True)
