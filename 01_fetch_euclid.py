# This code cross-matches SDSS & DESI catalogs with Euclid Q1 coverage, downloads thumbnail images of quasars from Euclid, and color-combines them to generate color thumbnail images. Takes ~12h (42000s) for downloading 4-filter (VIS/Y/J/H) imaging for ~3300 quasars. (260415)

# 260506: Cleaned up for Github upload

import os
import sys
import math
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits

# Specific imports for this code
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.esa.euclid import Euclid
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.data import conf
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.visualization import make_lupton_rgb, ImageNormalize, ManualInterval, LinearStretch
from astroquery.esa.euclid import Euclid
from skimage.color import rgb2hsv, hsv2rgb


t0=time.time()

fitsdir='fits_im'
jpegdir='jpeg'
os.makedirs(fitsdir, exist_ok=True)
os.makedirs(jpegdir, exist_ok=True)

# ============================================= 1. Read & match DESI & SDSS quasar catalogs ==========================


# read quasar catalog fits files: 1. DESI, 1,647,483 quasars
with fits.open('agnqso_desi.fits') as hdul:
	header = hdul[0].header
	data = hdul[1].data
	spectype = data['SPECTYPE']
	zwarn = data['ZWARN']
	program = data['PROGRAM']
	ra = data['TARGET_RA']
	dec = data['TARGET_DEC']
	redshift = data['Z']

# select QSOs
qsoidx_ = np.where((spectype == 'QSO') & (zwarn < 4))[0]
# remove duplicates
_, nondup = np.unique(data[qsoidx_]['TARGETID'], return_index=True)
qsoidx_desi=qsoidx_[nondup]

ra_desi, dec_desi = ra[qsoidx_desi], dec[qsoidx_desi]
z_desi = redshift[qsoidx_desi]

# read quasar catalog fits files: 2. SDSS DR16Q, 750,414 quasars
with fits.open('DR16Q_v4.fits') as hdul:
	header=hdul[0].header
	data=hdul[1].data
	ra=data['RA']
	dec=data['DEC']
	
ra_sdss, dec_sdss = ra, dec

print(f"# of quasars: {len(ra_desi)} (DESI) and {len(ra_sdss)} (SDSS)")


# =========================================== 2. Euclid Query ===============================================

###################### 2.1 Combined Euclid Q1 coverage map #######################

# combine Euclid Q1 HEALPix coverage maps
map_list = glob.glob("coverage_maps/*.fits", recursive=True)
all_valid_pixels = []
nside = None
is_nested = False

# these two lines should ideally be useless; list should be empty and no warnings generated
corrupted_files = []
warnings.simplefilter('ignore', category=AstropyUserWarning)

for k, f in enumerate(map_list):
	try:
		with fits.open(f) as hdul:
		# grab the HEALPix mapping rules from the header of the first file
			if nside is None:
				nside = hdul[1].header['NSIDE']
				ordering = hdul[1].header.get('ORDERING', 'RING')
				is_nested = (ordering.strip().upper() == 'NESTED')
			
			data = hdul[1].data
			pixels = data['PIXEL']
			weights = data['WEIGHT']
			
			# keep only pixels with actual coverage; weight = 0 means masked/empty, weight = 1 means fully unmasked
			valid_mask = weights > 0 
			all_valid_pixels.append(pixels[valid_mask])
	except Exception as e:
		corrupted_files.append(f)

if all_valid_pixels:
	master_pixel_array = np.unique(np.concatenate(all_valid_pixels))
	print(f"Total unique covered HEALPix pixels: {len(master_pixel_array)}")
else:
	print("No valid data found to combine.")

# report broken files
if corrupted_files:
    print(f"Found {len(corrupted_files)} intrinsically bad or corrupted mask files.")


###################### 2.2 Find quasars within Euclid Q1 coverage #######################

# convert quasar coordinates to healpix
qso_pixels_desi = hp.ang2pix(nside, ra_desi, dec_desi, lonlat=True, nest=is_nested)
qso_pixels_sdss = hp.ang2pix(nside, ra_sdss, dec_sdss, lonlat=True, nest=is_nested)

desi_euclid = np.isin(qso_pixels_desi, master_pixel_array)
sdss_euclid = np.isin(qso_pixels_sdss, master_pixel_array)

ra_desi_euclid = ra_desi[desi_euclid]
dec_desi_euclid = dec_desi[desi_euclid]
z_desi_euclid = z_desi[desi_euclid]

print(f"# of quasars in Euclid Q1: {np.sum(desi_euclid)} (DESI), {np.sum(sdss_euclid)} (SDSS)")

##################### 2.3 Download cutout .fits files & color-combine thumbnail images #####################

def apply_mtf_scaling(image_data, target_mean=0.2):
    """
    Applies Midtone Transfer Function (MTF) scaling for high-contrast visualization,
    as described in the Euclid Q1 Strong Lensing Discovery Engine A paper (Walmsley+25)
    """
    vmax = np.percentile(image_data, 99.85)
    vmin = np.min(image_data)
    norm_img = np.clip((image_data - vmin) / (vmax - vmin), 0, 1)
    
    x_0 = np.mean(norm_img)
    m = (x_0 * (1 - target_mean)) / (x_0 - 2 * x_0 * target_mean + target_mean)
    
    epsilon = 1e-8
    mtf_img = ((m - 1) * norm_img) / ((2 * m - 1) * norm_img - m + epsilon)
    return np.clip(mtf_img, 0, 1)



radius=5/3600.	# 5 arcsec radius
cutout_radius = 5.0 * u.arcsec
#for i in range(len(ra_desi_euclid)):
for i in range(3000,3001):
	ra0, dec0 = ra_desi_euclid[i], dec_desi_euclid[i]
	filterlist=['VIS','Y','J','H']
	for filtername in filterlist:
		inst = 'VIS' if filtername == 'VIS' else 'NISP'
		query = f"SELECT file_name, file_path, datalabs_path, instrument_name, filter_name, ra, dec, creation_date, product_type, patch_id_list, tile_index FROM q1.mosaic_product WHERE instrument_name='{inst}'"
		if inst == 'NISP':	# filter name addon only for NISP
			query += f" AND filter_name='NIR_{filtername}'"
		query += f" AND INTERSECTS(CIRCLE({ra0:.5f}, {dec0:.5f}, {radius:.5f}), fov)=1 "
		res = Euclid.launch_job_async(query).get_results()
		
		directory = res['file_path'][0]
		filename = res['file_name'][0]
		full_path = f"{directory}/{filename}"
		target_instrument = res['instrument_name'][0] 
		target_id = res['tile_index'][0] 
		print(f"Requesting cutout from: {full_path}")
		
		coord = SkyCoord(ra=ra0, dec=dec0, unit=(u.deg, u.deg))
		
		local_path = Euclid.get_cutout(file_path=full_path, 
			instrument=target_instrument, 
			id=str(target_id), 
			coordinate=coord, 
			radius=cutout_radius, 
			output_file=f"{fitsdir}/{i:04d}_{filtername}.fits", 
			verbose=False
		)


	##################### color-combine thumbnail images #####################


	im_vis = fits.getdata(f"{fitsdir}/{i:04d}_VIS.fits")
	im_y = fits.getdata(f"{fitsdir}/{i:04d}_Y.fits")
	im_j = fits.getdata(f"{fitsdir}/{i:04d}_J.fits")
	_, med_y, _ = sigma_clipped_stats(im_y)
	_, med_j, _ = sigma_clipped_stats(im_j)
	_, med_vis, _ = sigma_clipped_stats(im_vis)
	mtf_y = apply_mtf_scaling(im_y)
	mtf_j = apply_mtf_scaling(im_j)
	mtf_vis = apply_mtf_scaling(im_vis)

	if np.shape(im_vis) ==  np.shape(im_y) == np.shape(im_j):
		fwhm_kernel=np.sqrt(5**2-2**2)	# degrade VIS to NISP resolution
		kernel=Gaussian2DKernel(fwhm_kernel/2.355)
		vis_degrade=convolve(im_vis,kernel)
		image1 = make_lupton_rgb(im_j*0.5, im_y*0.8, vis_degrade*200, minimum=[med_vis, med_y, med_j], stretch=10, Q=5)
		image2 = np.dstack((mtf_j, mtf_y, mtf_vis)) # MTF
		image1_float = image1.astype(float) / 255.0
		vis_submed = im_vis - med_vis
		vis_submed[vis_submed<0] = 0 # prevent negative artifacts
		norm0=ImageNormalize(vis_submed, interval=ManualInterval(vmin=-0.01, vmax=0.05))
		l_channel=norm0(vis_submed)
		l_channel=np.clip(l_channel,0,1)	# ensure btw 0 and 1
		hsv_image = rgb2hsv(image1_float)
		hsv_image[...,2]=l_channel
		lrgb_final=hsv2rgb(hsv_image)
	
	plt.close('all')
	fig,axes=plt.subplots(2,2, figsize=(8,8), layout='constrained', gridspec_kw={'wspace': 0, 'hspace': 0})
	axes=axes.flatten()
	axes[0].imshow(image1, origin='lower')	# normal RGB
	axes[0].text(0.05, 0.95, "arcsinh VIS/Y/J, degraded", transform=axes[0].transAxes, fontsize=12, color='yellow', va='top', ha='left', fontweight='bold')
	norm1 = ImageNormalize(im_vis, interval=ManualInterval(vmin=-0.01, vmax=0.05), stretch=LinearStretch())
	axes[1].imshow(im_vis, cmap='gray', norm=norm1, origin='lower')	# origin='lower' is to reverse the jpg image so that it looks like the fits
	axes[1].text(0.05, 0.95, "VIS only", transform=axes[1].transAxes, fontsize=12, color='yellow', va='top', ha='left', fontweight='heavy')
	axes[2].imshow(image2, origin='lower')	# MTF
	axes[2].text(0.05, 0.95, "MTF VIS/Y/J", transform=axes[2].transAxes, fontsize=12, color='yellow', va='top', ha='left', fontweight='bold')
	axes[3].imshow(lrgb_final, origin='lower') # LRGB
	axes[3].text(0.05, 0.95, "LRGB arcsinh VIS+VIS/Y/J", transform=axes[3].transAxes, fontsize=12, color='yellow', va='top', ha='left', fontweight='bold')
	axes[1].text(0.85,0.95,f"{i:04d}", transform=axes[1].transAxes, fontsize=12, color='yellow', va='top', ha='left', fontweight='heavy')
	for ax in axes:
		ax.axis('off')
	plt.savefig(f"{jpegdir}/{i:04d}.jpg", dpi=300, bbox_inches='tight')
	plt.close('all')
	if i % 5 == 0:
		print(f"====={i}/{len(ra_desi_euclid)} finished, {time.time()-t0:.2f}s elapsed, {(time.time()-t0)/(i+1)*len(ra_desi_euclid)-(time.time()-t0):.2f}s left")
	

print(f"Execution time: {time.time()-t0:.2f}s")
