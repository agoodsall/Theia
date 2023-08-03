#code to produce an image giving the mu-psi coordinates for sharps using the near-real-time data from SDO


#===imports
import drms
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from skimage import measure
import math
from scipy.ndimage import gaussian_filter
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
plt.rcParams["figure.autolayout"] = True


#===Instantiate
c = drms.Client(email = "acgoodsall@gmail.com",verbose=False)
sm = c.query('hmi.Ic_noLimbDark_720s_nrt[$]', seg='Continuum') #this part isn't case sensitive
url_m = 'http://jsoc.stanford.edu' + sm.Continuum[0] #this part IS case sensitive
print("data retrieved from:", url_m)
image_data = fits.getdata(url_m)

#===Functions
def findcutoffs(arr_in):
	cut_off = 0
	count = 0
	nan_lines = []
	for i in np.arange(len(arr_in)):
		if i != count:
			cut_off = i
			break
		res = all(str(ele) == 'nan' for ele in arr_in[i]) #checks if the whole line is nans
		if res==True: #if it is all nans, it increases the cut-off margin to include that line
			nan_lines.append(i)
			count+=1
	return cut_off


def GetMuPhi(x,y, arr): #arr should be new_data

	r_pix = (len(arr)+len(arr[0]))/4
	x_dash = x-r_pix
	y_dash = r_pix-y
	r = np.sqrt((abs((x_dash)))**2+(abs((y_dash)))**2)
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

#===Main Code

#cropping the excess in the image
image_data_trans = image_data.T
cut_off_top = findcutoffs(image_data) - 1
cut_off_bottom = len(image_data) - findcutoffs(np.flipud(image_data))
cut_off_left = findcutoffs(image_data_trans) - 1
cut_off_right = len(image_data_trans) - findcutoffs(np.flipud(image_data_trans))
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


#===Finding the centre (the darkest point) of each spot
intensity_limit = 0.89
contours = measure.find_contours(new_data, intensity_limit*mean) #uses the 'marching squares' method
#https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py


fig, ax = plt.subplots()
ax.imshow(new_data, cmap=plt.cm.gray)

x_cens = []
y_cens = []


print("shape of new data", np.shape(new_data))
for e, contour in enumerate(contours):
	# if statement removes very small isolated points and the sun-is-a-spot issue
	if len(contour)>100 and len(contour)<2000:

		low  = int(min(contour[:,1]))
		upp  = int(max(contour[:,1]))
		left = int(min(contour[:,0]))
		right= int(max(contour[:,0]))

		indiv_sharp = new_data[left:right,low:upp]
		blur_geom_med = gaussian_filter(indiv_sharp, 2)

		y_cen_spot,x_cen_spot = find_min_idx(blur_geom_med)

		y_cen = y_cen_spot+left
		x_cen = x_cen_spot+low
		print("centres", x_cen, y_cen)
		x_cens.append(x_cen)
		y_cens.append(y_cen)

		mu_con, phi_con = GetMuPhi(x_cen, y_cen, new_data)
		
		ax.scatter(x_cen, y_cen, color="pink", s=20)
		ax.scatter(low, left, color = "orange", s=20)
		ax.scatter(upp, right, color = "orange", s=20)
		ax.scatter(x_cen, y_cen, color="white", s=10)
		if str(mu_con)!="nan":
			m=mu_con
			p=np.rad2deg(phi_con)
			ax.annotate("%.2f, %.2f"%(m, p), (x_cen, y_cen))
		else:
			ax.annotate("off disk, mu undefined", (x_cen, y_cen))

ax.scatter(0,0,color="white", s=30)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

#things to add:
#1. make it a class so it can be imoported easily by others
#2. make it find the magnetic field strength
#3. add just numbers to the image, then print out the mu and phi to the terminal
#4. add in a log file that has the time and the mu and phi values
#5. add in the predictive mode that includes the offset between the time at which the image was taken and the time the script is run