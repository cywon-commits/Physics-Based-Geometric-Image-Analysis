"""Image-segmentation feasibility only: user-provided nanoparticle montage is NOT license-cleared for publication.
Brick CC0 raster has no annotated reference count; this example is exploratory, not a validated accuracy claim.
"""
from pathlib import Path
import csv,numpy as np
from PIL import Image
import scipy.ndimage as ndi
from skimage import filters,morphology,measure,feature,segmentation,data
OUT=Path(__file__).resolve().parents[1]
rows=[]
# Small central ROI and size filtering, no learned thresholds
panel_img=np.asarray(Image.open('/mnt/data/image(2).png').convert('L'))/255
H,W=panel_img.shape;h=H//2;w=W//3
for k in range(6):
    rr=k%2;cc=k//2
    y0,y1=(rr*h+12,(rr+1)*h-14);x0,x1=(cc*w+14,(cc+1)*w-14)
    # skip last bottom if exact screenshot boundaries
    a=panel_img[y0:y1,x0:x1]
    a=ndi.gaussian_filter(a,1.1)
    cutoff=float(filters.threshold_otsu(a))
    mask=a>cutoff
    mask=morphology.remove_small_objects(mask,min_size=9)
    mask=ndi.binary_fill_holes(mask)
    dist=ndi.distance_transform_edt(mask)
    peaks=feature.peak_local_max(dist,min_distance=5,threshold_abs=2.3,labels=mask)
    markers=np.zeros_like(mask,dtype=np.int32)
    for j,(y,x) in enumerate(peaks):markers[y,x]=j+1
    labels=segmentation.watershed(-dist,markers,mask=mask)
    # boundary-crossing particle omitted from count, and very small isolated fragments
    regions=measure.regionprops(labels)
    cnt=sum(r.area>=12 and all(t>3 for t in (r.bbox[0],r.bbox[1],a.shape[0]-r.bbox[2],a.shape[1]-r.bbox[3])) for r in regions)
    rows.append({'sample':f'gold_panel_{"ace"[cc] if rr==0 else "bdf"[cc]}','source':'user-provided internet screenshot, NOT publication-cleared','provisional_watershed_count':int(cnt),
                 'threshold':cutoff,'ROI_width':a.shape[1],'ROI_height':a.shape[0]})
# CC0 brick photo; explicit count is approximate and not "true brick count"
a=np.asarray(data.brick(),dtype=np.float32)/255
mask=ndi.gaussian_filter(a,.9)<filters.threshold_otsu(a)
mask=morphology.binary_opening(mask,morphology.disk(1))
labels,n=ndi.label(mask)
regions=measure.regionprops(labels)
count=sum(40<=r.area<=4000 and r.eccentricity>.5 and all(t>2 for t in (r.bbox[0],r.bbox[1],a.shape[0]-r.bbox[2],a.shape[1]-r.bbox[3])) for r in regions)
rows.append({'sample':'CC0 brick image','source':'skimage.data.brick (CC0)','provisional_watershed_count':int(count),
             'threshold':float(filters.threshold_otsu(a)),'ROI_width':a.shape[1],'ROI_height':a.shape[0]})
with open(OUT/'data'/'unvalidated_image_counts.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(r['sample'],r['provisional_watershed_count'])
