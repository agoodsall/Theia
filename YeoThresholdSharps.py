
#===imports
import drms
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import cv2
import imutils
from skimage import measure
from geomtric_mean_optimisation import geometricMedian, distSum, Point
from scipy.signal import argrelmin, argrelmax
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["figure.autolayout"] = True
import matplotlib

#===Instantiate
c = drms.Client(email = "acgoodsall@gmail.com",verbose=False)
sm = c.query('hmi.Ic_noLimbDark_720s_nrt[$]', seg='Continuum') #this part isn't case sensitive
url_m = 'http://jsoc.stanford.edu' + sm.Continuum[0] #this part IS case sensitive
print("data retrieved from:", url_m)
image_data = fits.getdata(url_m)

#maybe loop isn't needed?
cut_off_top = 0
cut_off_bottom = 0

def findcutoffs(arr_in):
	cut_off = 0
	count = 0
	nan_lines = []
	for i in np.arange(len(arr_in)):
		if i != count:
			cut_off = i
			break
		res = all(str(ele) == 'nan' for ele in arr_in[i]) #checks if the whole line is nans
		if res==True:
			nan_lines.append(i)
			count+=1
	return cut_off

image_data_trans = image_data.T
cut_off_top = findcutoffs(image_data) - 1
cut_off_bottom = len(image_data) - findcutoffs(np.flipud(image_data))
cut_off_left = findcutoffs(image_data_trans) - 1
cut_off_right = len(image_data_trans) - findcutoffs(np.flipud(image_data_trans))

# plt.imshow(image_data)
# plt.hlines(cut_off_top, 0,2048)
# plt.hlines(cut_off_bottom, 0,2048)
# plt.vlines(cut_off_left, 0,2048)
# plt.vlines(cut_off_right, 0,2048)
# plt.show()

truncated_data = image_data[cut_off_top:cut_off_bottom]
trans_trunc_data = truncated_data.T[cut_off_left:cut_off_right]
new_data = trans_trunc_data.T

#now to calculate the mean you do need to get rid of all nans:
new_data_no_nans = new_data
count_data = 0
for i in np.arange(0,len(new_data),1):
	for j in np.arange(0,len(new_data[0]),1):
		if str(new_data[i][j]) == 'nan':
			new_data_no_nans[i][j] = 0
			count_data+=1

n2 = len(new_data[0])*len(new_data)
n_data_points = n2 - count_data

mean = np.mean(new_data_no_nans)*(n2/n_data_points) #over flattened array


# Z = new_data*0.
# for i in np.arange(0,len(new_data),1):
# 	for j in np.arange(0,len(new_data[0]),1):
# 		if abs(new_data[i][j]-mean) <= 0.1:
# 			Z[i][j] = 0.89



# X = []
# Y = []

# X = np.linspace(0, 360, len(image_data[0]))
# Y = np.linspace(0, 360, len(image_data))

# Z = image_data

# x, y = np.meshgrid(X, Y)

# fig, ax = plt.subplots(figsize=(6, 6))
# levels = [0,0.5*mean, np.amax(new_data_no_nans)]
# CS = ax.contour(X, Y, Z, levels, cmap='bwr')
# plt.colorbar(CS)
# ax.set_ylabel('sq Latitude', fontsize=22)
# ax.set_xlabel('sq Longitude', fontsize=22)
# im = plt.pcolormesh(x, y, image_data, cmap='gray', shading = 'nearest')
# # plt.clim(0.9,1.65)
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# # # plt.title('Contours_2042')
# plt.savefig('Contours_Yeo_200723')
# plt.show()

#find below 0.5, for umbra:
rows, cols = np.where(new_data-0.2 <= 0.1)

# plt.imshow(new_data)
# plt.scatter(cols,rows, color="red", s=5)
# plt.show()


#===connected component analysis
intensity_limit = 0.89
contours = measure.find_contours(new_data, intensity_limit*mean) #uses the 'marching squares' method
#https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py

# for f, c in enumerate(contours[0:19]):
# 	geom_med_arr = []
# 	print("here", len(c), len(c[:,1]))
# 	if len(c[:,1])>100:
# 		left  = int(min(c[:,1]))
# 		print("contour x values", c[:,1])
# 		right = int(max(c[:,1]))
# 		low   = int(min(c[:,0])) #remember it's a fits file - top left
# 		upp   = int(max(c[:,0]))
# 		print("edges", left, right, low, upp)
# 		geom_med_arr = new_data[left:right,low:upp]
# 		contour1_centrex, contour1_centrey = geometricMedian(geom_med_arr,30)
# 		print("centre", contour1_centrex, contour1_centrey)

fig, ax = plt.subplots()
ax.imshow(new_data, cmap=plt.cm.gray)
# plt.show()


geom_arr = []
x_cens = []
y_cens = []

def GetMuPhi(x,y, arr): #arr should be new_data
	# print(x[i], y[i])
	r_pix = (len(arr)+len(arr[0]))/4
	x_dash = x-r_pix
	y_dash = r_pix-y
	r = np.sqrt((abs((x_dash)))**2+(abs((y_dash)))**2)
	# print("pixel radius", r_pix)
	# mu = np.cos(r*np.pi/(2*r_pix))
	mu = np.sqrt(1-(r/r_pix)**2)

	theta = math.atan(abs(y_dash)/abs(x_dash))
	phi = None
	if x_dash>=0 and y_dash>=0:
		phi = np.pi/2 - theta
	if x_dash>=0 and y_dash<=0:
		phi = np.pi/2 + theta
	if x_dash<=0 and y_dash<=0:
		phi = 2*np.pi - (theta+np.pi/2)
	if x_dash<=0 and y_dash>=0:
		phi = 2*np.pi - theta

	return mu, phi

def find_min_idx(arr):
    k = arr.argmin()
    ncol = arr.shape[1]
    return k/ncol, k%ncol #row then column

print("shape of new data", np.shape(new_data))
for e, contour in enumerate(contours):
	if len(contour)>100 and len(contour)<2000:
		print(e)
		# fig, ax = plt.subplots(1,2)
		# ax[0].imshow(new_data, cmap=plt.cm.gray)
		# ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2, label="%f" %(e))

		low  = int(min(contour[:,1]))
		upp  = int(max(contour[:,1]))
		left = int(min(contour[:,0])) #remember it's a fits file - top left
		right= int(max(contour[:,0]))

		geom_med_arr = new_data[left:right,low:upp]
		blur_geom_med = gaussian_filter(geom_med_arr, 2)
		print(np.shape(blur_geom_med))

		y_cen_spot,x_cen_spot = find_min_idx(blur_geom_med)
		
		# print(x_cen, y_cen)

		# fig,axs = plt.subplots(1,2)
		# axs[0].imshow(geom_med_arr)
		# axs[1].imshow(gaussian_filter(geom_med_arr, 2))
		# axs[1].scatter(x_cen, y_cen, color="white", s=10)
		# plt.show()

		y_cen = y_cen_spot+left
		x_cen = x_cen_spot+low
		print("centres", x_cen, y_cen)
		x_cens.append(x_cen)
		y_cens.append(y_cen)

		mu_con, phi_con = GetMuPhi(x_cen, y_cen)
		
		ax.scatter(x_cen, y_cen, color="pink", s=20)
		ax.scatter(low, left, color = "orange", s=20)
		ax.scatter(upp, right, color = "orange", s=20)
		ax.scatter(x_cen, y_cen, color="white", s=10)
		ax.annotate("%f %f"%(mu_con, np.rad2deg(phi_con)), (x_cen, y_cen))
		# ax.imshow(geom_med_arr, cmap=plt.cm.gray)

ax.scatter(0,0,color="white", s=30)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()