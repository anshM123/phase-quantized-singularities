"""Seed original stored IPM profiles into the original independent global sparse solver.
This is a cross-formulation numerical test, not a proof or continuum-convergence certificate.
"""
import sys,time,json,traceback,gc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ncs-submission/code/lib"))
from ipm_global import IPMGlobal
CASES=[
 dict(label="n0",file="ipm_profile_n0_lam1.0285722975_Nb32_hs0.025.npy",lam=1.0285722975,source_hs=.025,grids=[(-8,10,.20,12),(-9,12,.15,16)]),
 dict(label="z7_state",file="ipm_state_z7.3460_Nb32_hs0.00625.npy",lam=1/7.346,source_hs=.00625,grids=[(-8,10,.20,12),(-9,12,.15,16)])]
out=[]
for c in CASES:
 for sm,sx,hs,nb in c["grids"]:
  t=time.time();job=f"{c['label']} sm{sm} sx{sx} hs{hs} nb{nb}"
  print("START",job,flush=True)
  try:
   G=IPMGlobal(s_min=sm,s_max=sx,hs=hs,Nb=nb)
   U0=G.initial_from_march(str(ROOT/"ncs-submission/data/profiles"/c["file"]),c["lam"],Nb_src=32,hs_src=c["source_hs"],s_start=-20)
   init=float(np.max(np.abs(G.residual(U0))))
   print("INITIAL",init,"size",len(U0),flush=True)
   U,ok=G.newton(U0,tol=1e-9,maxit=13,verbose=True)
   final=float(np.max(np.abs(G.residual(U))))
   R,Om,X,lam=G.unpack(U);D=1+lam+G.Dbeta@X
   row=dict(label=c["label"],s_min=sm,s_max=sx,hs=hs,Nb=nb,
      lambda_initial=c["lam"],lambda_computed=float(lam),
      z_computed=float(1/lam) if lam!=0 else None,
      initial_residual=init,final_residual=final,newton_ok=bool(ok),
      min_transport_speed=float(D.min()),
      difference_from_input_lambda=float(lam-c["lam"]),
      seconds=round(time.time()-t,1),
      verdict=("DISCRETE CONVERGED ONLY; NOT continuum verified" if ok and D.min()>0 else "UNVERIFIED"))
  except Exception as exc:
   row=dict(label=c["label"],grid=[sm,sx,hs,nb],error=str(exc),
       traceback=traceback.format_exc(),seconds=round(time.time()-t,1))
  out.append(row);print("GLOBAL_RESULT",json.dumps(row),flush=True)
  (ROOT/"validation/seeded_global_crosscheck_results.json").write_text(json.dumps(out,indent=2))
  gc.collect()
print("COMPLETE",flush=True)
