"""Known-angle physical calibration and NNLS inversion; no neural weights."""
from pathlib import Path
import argparse,csv,hashlib,io,zipfile
import numpy as np
from PIL import Image

def feature(raw,roi,side,exposure):
    with Image.open(io.BytesIO(raw)) as im:
        im=im.convert('RGB')
        if roi:
            a,b,c,d=roi
            if not (0<=a<c<=im.width and 0<=b<d<=im.height):raise ValueError('ROI outside frame')
            im=im.crop(roi)
        x=np.asarray(im.resize((side,side),Image.Resampling.BOX),dtype=float).ravel()
    return x*(30000./exposure/255.)/np.sqrt(x.size)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,default=Path('data'))
    ap.add_argument('--archives',type=Path,help='Read independent ZIP assets without extraction')
    ap.add_argument('--calibration-out',type=Path)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--image',type=Path)
    ap.add_argument('--roi',nargs=4,type=int)
    ap.add_argument('--side',type=int,default=32)
    ap.add_argument('--exposure-us',type=float,default=30000)
    a=ap.parse_args()
    if a.side<1 or a.exposure_us<=0:ap.error('Positive side and exposure required')
    if a.calibration_out:
        if a.calibration_out.exists():raise FileExistsError(a.calibration_out)
        with (a.data_root/'labels.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        rows.sort(key=lambda r:float(r['nominal_wavelength_nm']))
        assert len(rows)==101 and all(float(r['angle_deg'])==5 for r in rows)
        columns=[]
        for row in rows:
            if a.archives:
                with zipfile.ZipFile(a.archives/row['archive']) as z:raw=z.read(row['file'])
            else:raw=(a.data_root/row['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Data checksum mismatch')
            columns.append(feature(raw,a.roi,a.side,float(row['exposure_us'])))
        matrix=np.stack(columns,axis=1)
        np.savez_compressed(a.calibration_out,matrix=matrix,wavelength_nm=np.array([float(r['nominal_wavelength_nm']) for r in rows]),roi=np.array(a.roi or [],int),side=a.side,angle_deg=5)
        print(f'Calibrated {len(rows)} nominal wavelength bins at +5 degrees; matrix {matrix.shape}.')
    elif a.calibration and a.image:
        from scipy.optimize import nnls
        with np.load(a.calibration) as f:
            roi=f['roi'].tolist() or None;side=int(f['side']);mat=f['matrix'];waves=f['wavelength_nm']
        if a.roi is not None and a.roi!=roi:raise ValueError('ROI must match calibration')
        y=feature(a.image.read_bytes(),roi,side,a.exposure_us)
        scale=max(np.linalg.norm(y),1e-30)
        spectrum,_=nnls(mat,y/scale,maxiter=10000);spectrum*=scale
        print('nominal_wavelength_nm,coefficient')
        for w,v in zip(waves,spectrum):print(f'{w:g},{v:.10g}')
    else:ap.error('Specify --calibration-out, or --calibration together with --image')

if __name__=='__main__':main()
