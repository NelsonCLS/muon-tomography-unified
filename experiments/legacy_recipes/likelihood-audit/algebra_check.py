"""Differential test against functions extracted from the supplied original source."""
from pathlib import Path
from types import SimpleNamespace
import ast,json
import numpy as np
from muon_tomo.reconstruction import LikelihoodReconstructor
from muon_tomo.reconstruction.common import physical_grid_intersections
from muon_tomo.types import TrackTable

original=Path('/Users/longsangchan/Library/Mobile Documents/com~apple~CloudDocs/CUHK Ph.D/Muon Tomography/muonTruck/recon/utils_likelihood.py')
tree=ast.parse(original.read_text())
names={'update_S','update_lambda','update_S_angleOnly','update_lambda_angleOnly'}
functions=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(functions)==len(names)
ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(original),'exec'),ns)

# Actual production full-covariance method versus original update functions.
tracks=TrackTable(np.array([[0,0,50.]]*3),np.array([[0,0,-1.]]*3),np.array([[.1,-.1,-50.],[.4,.3,-50.],[-.3,.1,-50.]]),np.array([[.001,-.003,-1.],[.003,.002,-1.],[-.002,.001,-1.]]),np.array([2000.,3000.,5000.]))
r=LikelihoodReconstructor(iterations=1,epsilon=0.,angular_resolution_rad=0.,position_resolution_cm=0.,maximum_inverse_radiation_length_cm_inv=1e9,update_statistic='mean')
paths=[np.arange(4,dtype=np.int64)]*3
lens=[np.array([10.,20.,30.,40.]),np.array([25.,25.,25.,25.]),np.array([40.,30.,20.,10.])]
W=[]
for ll in lens:
    T=np.cumsum(ll[::-1])[::-1]-ll
    ww=np.empty((4,2,2));ww[:,0,0]=ll;ww[:,0,1]=ww[:,1,0]=ll**2/2+ll*T;ww[:,1,1]=ll**3/3+ll**2*T+ll*T**2;W.append(ww)
a,di,b,do=physical_grid_intersections(tracks,r.geometry)
tx=np.arctan(do[:,0]/abs(do[:,2]));ty=np.arctan(do[:,1]/abs(do[:,2]))
# For vertical incidence Eq.14 cosine factors cancel exactly.
Dx=np.stack((tx,b[:,0]-a[:,0]),1)[:,:,None];Dy=np.stack((ty,b[:,1]-a[:,1]),1)[:,:,None]
scale=(15/3000)**2
old=SimpleNamespace(p_r2=(3000/tracks.momentum_mev)**2,D_x=Dx,D_y=Dy,passed_voxels=paths,W=W,lambdaa=np.full(4,4e-5*scale),S_total=np.zeros(4),times_being_hit=np.zeros(4),total_voxels=4)
ns['update_S'](old);ns['update_lambda'](old)
current=r._full_covariance_mlem(tracks,tracks.momentum_mev,paths,lens)[0][:4]
np.testing.assert_allclose(current,old.lambdaa/scale,rtol=1e-12,atol=1e-15)

# Single unknown voxel, two unequal-length rays: known Gaussian variance MLE.
q=np.array([.001,.1]);length=np.array([1.,10.])
dummy=SimpleNamespace(projected_scattering_angles=lambda:(np.sqrt(q),np.sqrt(q)))
ar=LikelihoodReconstructor(iterations=30)
angular=ar._angular_mlem(dummy,np.full(2,15.),[np.array([0])]*2,[ll[None] for ll in length])[0][0]
angle_old=SimpleNamespace(p_r2=np.ones(2),D_x=np.sqrt(q)[:,None,None],D_y=np.sqrt(q)[:,None,None],passed_voxels=[np.array([0])]*2,W=[np.array([[[ll,0.],[0.,1.]]]) for ll in length],lambdaa=np.array([4e-5]),S_total=np.zeros(1),times_being_hit=np.zeros(1),total_voxels=1)
for _ in range(60):ns['update_S_angleOnly'](angle_old);ns['update_lambda_angleOnly'](angle_old)
gaussian=float(np.mean(q/length));ratio=float(q.sum()/length.sum())
np.testing.assert_allclose(angular,ratio)
np.testing.assert_allclose(angle_old.lambdaa[0],gaussian)
assert not np.isclose(angular,gaussian)
def nll(rho):return float(np.sum(np.log(rho*length)+q/(rho*length)))
result={'full_covariance_one_iteration':{'maximum_absolute_rho_difference':float(np.max(abs(current-old.lambdaa/scale))),'original':(old.lambdaa/scale).tolist(),'current':current.tolist(),'conditions':'matched observations/paths/momenta, mean update, error covariance and clipping disabled'},'angular_one_voxel':{'q':q.tolist(),'length_cm':length.tolist(),'current_update':float(angular),'original_angleOnly':float(angle_old.lambdaa[0]),'analytic_gaussian_MLE':gaussian,'current_gaussian_nll':nll(angular),'optimal_gaussian_nll':nll(gaussian)}}
(Path(__file__).parent/'algebra_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
