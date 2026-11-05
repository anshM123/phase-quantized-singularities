"""Branch-constrained, higher-resolution seeded global IPM experiment.
Independent global formulation from original saved marching states.
No continuum-convergence or rigorous solution certification is claimed.
"""
from pathlib import Path
import sys,time,json,traceback,gc
import numpy as np
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ncs-submission/code/lib"))
from ipm_global import IPMGlobal
CASES=[
 ("n0","ipm_profile_n0_lam1.0285722975_Nb32_hs0.025.npy",1.0285722975,.025,.075,24),
 ("z7","ipm_state_z7.3460_Nb32_hs0.00625.npy",1/7.346,.00625,.075,24),
 ("z7","ipm_state_z7.3460_Nb32_hs0.00625.npy",1/7.346,.00625,.05,32)]
def newton(G,U,target,tolerance,steps=16):
 history=[]
 for k in range(steps):
  residual=G.residual(U)
  err=float(np.max(np.abs(residual)))
  lam=float(U[-1])
  D=1+lam+(G.Dbeta@U[2*G.N:3*G.N])
  mn=float(D.min())
  row=dict(iter=k,residual=err,lambda_=lam,min_D=mn)
  history.append(row);print("ITER",json.dumps(row),flush=True)
  if err<1e-9 and mn>0:return U,True,history
  try: step=splu(G.jacobian(U)).solve(-residual)
  except Exception as e:row["failure"]=str(e);break
  found=False
  for j in range(15):
   t=2**(-j);trial=U+t*step
   if abs(trial[-1]-target)>tolerance:continue
   Dtrial=1+trial[-1]+(G.Dbeta@trial[2*G.N:3*G.N])
   if np.min(Dtrial)<=0:continue
   nxt=float(np.max(np.abs(G.residual(trial))))
   if nxt<err*(1-1e-4*t):U=trial;found=True;break
  if not found:row["failure"]="branch constraint/positive transport/line search failure";break
 return U,False,history
out=[]
for label,filename,lam,source_hs,hs,nb in CASES:
 t=time.time();print("CASE",label,"hs",hs,"nb",nb,flush=True)
 try:
  G=IPMGlobal(s_min=-10,s_max=14,hs=hs,Nb=nb)
  U=G.initial_from_march(str(ROOT/"ncs-submission/data/profiles"/filename),lam,Nb_src=32,hs_src=source_hs,s_start=-20)
  initial=float(np.max(np.abs(G.residual(U))))
  U,ok,history=newton(G,U,lam,tolerance=.035 if label=="n0" else .006)
  row=dict(label=label,hs=hs,Nb=nb,target_lambda=lam,computed_lambda=float(U[-1]),delta_lambda=float(U[-1]-lam),initial_residual=initial,final_residual=float(np.max(abs(G.residual(U)))),converged=ok,history=history,seconds=round(time.time()-t,1))
 except Exception as e:row=dict(label=label,hs=hs,Nb=nb,error=str(e),traceback=traceback.format_exc())
 out.append(row)
 print("HIRES_RESULT",json.dumps({k:v for k,v in row.items() if k!="history"}),flush=True)
 (ROOT/"validation/seeded_global_highres_results.json").write_text(json.dumps(out,indent=2))
 gc.collect()
print("DONE",flush=True)
