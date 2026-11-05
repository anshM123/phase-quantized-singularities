"""Cross-discretization PDE residual audit of released IPM binary states.
This does NOT prove an independently converged singularity profile.
"""
import gc, json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"ncs-submission/code/lib"))
from ipm_solver import IPM
from bq_newton import full

CASES=[
 ("n0","ipm_profile_n0_lam1.0285722975_Nb32_hs0.025.npy",1.0285722975,.025),
 ("n6","ipm_profile_n6_lam0.15092_Nb32_hs0.0125.npy",.15092,.0125),
 ("z7_state","ipm_state_z7.3460_Nb32_hs0.00625.npy",1/7.346,.00625)]
def stats(a):
 a=abs(np.asarray(a));a=a[np.isfinite(a)]
 return dict(max=float(a.max()),p95=float(np.percentile(a,95)),rms=float(np.sqrt(np.mean(a*a))))
def audit(label,name,lam,hs):
 t=time.time()
 B=IPM(lam,Nb=32,hs=hs,s_sw=12,s_start=-20,s_min=-120,s_max=100)
 Y=np.load(R/"ncs-submission/data/profiles"/name)
 X=full(B,Y)
 assert X.shape==(B.Ns,B.Nb+1),(X.shape,B.Ns,B.Nb+1)
 v=B.march(X/B.ea2[:,None],return_all=True)
 m=float(v["m"]);A=float(v["A"])
 sel=(B.s>=-9)&(B.s<=6)
 s=B.s[sel];Xs=X[sel];Rho=v["Th"][sel]*np.clip(B.cb,0,None)[None,:]**m;Om=v["Omega"][sel]
 deriv=lambda a,k:savgol_filter(a,9,8,deriv=k,delta=hs,axis=0,mode="interp")
 Xsd=deriv(Xs,1);Xssd=deriv(Xs,2);Rsd=deriv(Rho,1)
 Xb=Xs@B.Db.T;Xbb=Xs@B.Dbb.T;Rb=Rho@B.Db.T
 D=1+lam+Xb;w=-(2*Xs+Xsd)
 rt=-lam*Rho+D*Rsd+w*Rb
 rp=Xssd+4*Xsd+4*Xs+Xbb+Om
 ro=Om+np.exp(-s)[:,None]*(B.cb[None,:]*Rsd-B.sb[None,:]*Rb)
 mid=(s>=-8)&(s<=5);interior=np.ix_(mid,np.arange(1,B.Nb))
 result=dict(case=label,lambda_=float(lam),m=m,m_minus_2=m-2,A=A,
    D0=float(1+lam-A),min_transport_speed=float(D.min()),
    transport_residual=stats(rt[interior]),poisson_residual=stats(rp[interior]),
    slaving_residual=stats(ro[interior]),
    scaled_transport_residual=stats((rt/(1+abs(lam*Rho)+abs(D*Rsd)+abs(w*Rb)))[interior]),
    scaled_poisson_residual=stats((rp/(1+abs(Xssd)+abs(4*Xsd)+abs(4*Xs)+abs(Xbb)+abs(Om)))[interior]),
    elapsed_seconds=round(time.time()-t,1),
    note="Independent finite-difference residual of archived original solution. Not an independently converged nonlinear solve.")
 del B,Y,X,v;gc.collect()
 return result
if __name__=="__main__":
 ans=[]
 for c in CASES:
  try: result=audit(*c)
  except Exception as e:result=dict(case=c[0],error=str(e),traceback=traceback.format_exc())
  print("AUDIT_RESULT",json.dumps(result),flush=True)
  ans.append(result)
 p=R/"validation"/"independent_profile_audit_results.json"
 p.write_text(json.dumps(ans,indent=2))
 print("FINISHED",p,flush=True)
