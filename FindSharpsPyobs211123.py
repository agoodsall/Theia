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
from astropy.time import Time
import numpy.ma as ma
from scipy.interpolate import interpn
from skimage.transform import resize
import csv
import math
import urllib.request
import urllib.error
import csv



plt.style.use(astropy_mpl_style)
plt.rcParams["figure.autolayout"] = True


class FindSharps():
	def __init__(self, mode, Predict=True, Verbose=True, Static=True, Dynamic=False):
		IntensityLimit = 0.89

		#query and download the data
		self.data_i, self.data_m = self.Instantiate()
		#crop the data to the same size in pixels to remove nans at the edge
		cropdata_i = self.CropNans(self.data_i, "nan")
		cropdata_m_pre = self.CropNans(self.data_m, "nan")
		
		#set up coordinate grid for Mu on the magnetic field uncorrected image.
		X_pre = np.arange(len(cropdata_m_pre[0]))
		Y_pre = np.arange(len(cropdata_m_pre))
		Z_pre = np.meshgrid(X_pre,Y_pre)
		Mugrid_pre,phi_pre = self.GetMuPhi2D(Z_pre)


		# plt.imshow(Mugrid)
		# plt.colorbar()
		# plt.show()

		#set up coordinate grid for intensity
		X_i = np.arange(len(cropdata_i[0]))
		Y_i = np.arange(len(cropdata_i))
		Z_i = np.meshgrid(X_i,Y_i)

		#mask the band at mu<0.3 in magnetic field
		cropdata_m_noband_etc = ma.masked_where(Mugrid_pre<0.3, cropdata_m_pre, copy=True)
		#mask any values that are nans
		cropdata_m_noband = ma.masked_where(np.isnan(Mugrid_pre)==True, cropdata_m_noband_etc, copy=True)
		#crop image to rows and columns without nans and therefore remove the band and <0.3
		cropdata_m = self.CropNans(cropdata_m_noband,"--")

		# plt.imshow(cropdata_m_noband)
		# plt.title("Magnetic field with 0.3 band masked")
		# plt.show()
		#==============
		# plt.imshow(cropdata_m)
		# plt.title("Magnetic field with 0.3 band cut")
		# plt.show()

		#re-set up the coordinate grid for rescaling the intensity
		X = np.arange(len(cropdata_m[0]))
		Y = np.arange(len(cropdata_m))
		Z = np.meshgrid(X,Y)

		Mugrid,Phigrid = self.GetMuPhi2D(Z)

		#rescale the intensity to the same as the magnetogram
		interped_i = resize(cropdata_i, (len(X),len(Y))) # default is linear interpolation

		#correct for foreshortening
		B_by_noise = cropdata_m*Mugrid
		B_by_noise_phi = cropdata_m*Phigrid
		# plt.imshow(B_by_noise_phi)
		# plt.title("Phi with masked active regions")
		# plt.show

		plt.imshow(interped_i)
		plt.show()

		#to calculate just the quiet mean, mask nans and magnetically active regions/pixels
		# mask nans
		# I_mask_nonans = ma.masked_where(np.isnan(B_by_noise)==True,interped_i)
		I_maskactive = interped_i
		I_maskquiet = B_by_noise
		# mask the magnetically active pixels
		a_eq1 = ma.masked_where(B_by_noise > 24,I_maskactive,copy=True) # 24 G/mu_ij - Haywood16 pot. should use flattened
		active_mask = ma.getmaskarray(a_eq1)
		# calculate the mean of the non-magnetically active pixels.
		#to make it clearer do a_eq1.data
		fig, axs = plt.subplots(1,2)
		axs[0].imshow(a_eq1)
		axs[1].imshow(active_mask)
		plt.show()
		print("shape of spot masked array", np.shape(a_eq1))
		QuietMean = np.nanmean(a_eq1)
		print("means", QuietMean)

		# plt.imshow(active_mask)
		# plt.title("intensity, masked nans and ARs")
		# plt.show()


		spotalias = I_maskactive
		facualias = I_maskactive
		Boolarrayspot = np.full(np.shape(interped_i),0)
		Boolarrayplage = np.full(np.shape(interped_i),0)
		Boolarrayspot[interped_i<=0.89*QuietMean] = 1
		plt.imshow(Boolarrayspot)
		plt.show()
		Boolarrayplage[interped_i>0.89*QuietMean] = 1
		plt.imshow(Boolarrayplage)
		plt.show()

		# high_intensity_1_eq3 = ma.masked_where(interped_i> 0.89,spotalias, copy=True) #*Quietmean? #spotalias needs to be interped_i
		# low_intensity_1_eq2  = ma.masked_where(interped_i<=0.89,spotalias, copy=True)
		eq1_bitmap = active_mask
		# eq2_bitmap = ma.getmaskarray(low_intensity_1_eq2)
		# eq3_bitmap = ma.getmaskarray(high_intensity_1_eq3)

		just_quiet = np.invert(eq1_bitmap)
		just_spots = eq1_bitmap*Boolarrayspot #eq2_bitmap
		just_plage = eq1_bitmap*Boolarrayplage #*eq3_bitmap

		full = just_quiet*2 + just_plage*3 + just_spots*4
		rng = np.random.default_rng()

		plt.imshow(full, cmap="tab10")
		plt.show()
		self.PlotDynamic(full)

		if mode=="Spot":
			centres_x_sp, centres_y_sp,plot_contours = self.FindPoints(full,interped_i, 3)
			mus, phis = self.FindPointsMuPhi(centres_x_sp,centres_y_sp,interped_i)
			#pick random spot:
			print(mus,phis)
			rand_idx = rng.integers(low=0, high=len(mus), size=1)
			mu_final = np.array([mus[rand_idx[0]]])
			phi_final = np.array([phis[rand_idx[0]]])
			if Predict==True:
				mup, phip = self.FindPredMuPhi(mu_final,phi_final)
				print("predicted", mup, phip)
			else:
				phi_final=np.rad2deg(phi_final)

		if mode=="Quiet":
			a = full==2
			quiet_mu = a*Mugrid
			quiet_phi = a*Phigrid
			quiet_mu[np.isnan(quiet_mu)==True] = 0
			quiet_phi[np.isnan(quiet_phi)==True] = 0
			print("phi shape", np.shape(quiet_phi), np.shape(quiet_mu))
			indices = np.nonzero(quiet_mu) #returned row-major. indices[0] is row index, indices[1] is column index
			#hopefully temporary - random choice:
			print("shape in quiet", np.shape(indices))
			rand_idx = rng.integers(low=0, high=len(indices[0]), size=1)
			row = indices[0][rand_idx][0]
			col = indices[1][rand_idx][0]
			print("row col in quiet:", row, col)
			mu_final = np.array([quiet_mu[row][col]])
			phi_final = np.array([quiet_phi[row][col]])

			if Predict==True:
				mups, phips = self.FindPredMuPhi(mu_final,phi_final)
				mup = mups[0]
				phip = phips[0]
				print("predicted", mup, phip)
			else:
				phi_final=np.rad2deg(phi_final)


		if mode=="Plage":
			a = full==3
			plt.imshow(a)
			plt.show()
			
			plage_mu = a*Mugrid
			plage_phi = a*Phigrid
			plage_mu[np.isnan(plage_mu)==True] = 0
			plage_phi[np.isnan(plage_phi)==True] = 0

			print("phi shape", np.shape(plage_phi), np.shape(plage_mu))
			indices = np.nonzero(plage_mu) #returned row-major. indices[0] is row index, indices[1] is column index
			#hopefully temporary - random choice:
			print("shape in plage", np.shape(indices))
			rand_idx = rng.integers(low=0, high=len(indices[0]), size=1)
			col = indices[0][rand_idx][0]
			row = indices[1][rand_idx][0]
			plt.imshow(plage_mu)
			plt.scatter(row,col,color="green")
			plt.show()
			print("row col in plage", row, col)
			print("shape of plage_mu", np.shape(plage_mu))
			mu_final = np.array([plage_mu[row][col]])
			phi_final = np.array([plage_phi[row][col]])
			print("mu and phi in plage", mu_final, phi_final)

			if Predict==True:
				mups, phips = self.FindPredMuPhi(mu_final,phi_final)
				print(mups, phips)

				mup = mups[0]
				phip = phips[0]
				print("predicted", mup, phip)
			else:
				phi_final=np.rad2deg(phi_final)
			

		#find which mus are close to a set obs mu value - within the tolerance 'limit'.
		# mu_obs_list_quiet,phi_obs_list_quiet = self.find_mu_min(mus_list_quiet,phis_list_quiet,limit=1e-7)
		# print("phis", phi_obs_list_quiet)

		with open('testmufromtheia.csv', 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',)
			spamwriter.writerow((mu_final,phi_final))

		# if Verbose==True:
		# 	self.PrintLog(mus, phis)


	def Instantiate(self,match_SDO=True, obs_datime = "$"):
		#===Instantiate
		c = drms.Client() #email = "acgoodsall@gmail.com",verbose=False
		time_obs = obs_datime #"2023.11.20_TAI"
		try:
			sm = c.query('hmi.Ic_noLimbDark_720s_nrt['+time_obs+']', seg='Continuum') #this part isn't case sensitive
			mag = c.query('hmi.M_720s_nrt['+time_obs+']', seg='magnetogram') #hmi.Mharp_720s_nrt #Marmask_720s_nrt #M_720s
			self.keys = c.query('hmi.Ic_noLimbDark_720s_nrt['+time_obs+']', key=('DATE__OBS','CROTA2'))
			#key_m = c.query('hmi.Marmask_720s_nrt[$]', key=('DATE__OBS'))
		except Exception as e:
			print("Exception", e)

		url_c = 'http://jsoc.stanford.edu' + sm.Continuum[0] #this part IS case sensitive
		url_m = 'http://jsoc.stanford.edu' + mag.magnetogram[0] #this part IS case sensitive
		
		print("data retrieved from:", url_c)
		image_data = fits.getdata(url_c)
		mag_data = fits.getdata(url_m)
			
		if match_SDO==True:
			image_data = np.rot90(image_data,2) #SDO HMI is upside down (like 180.02 degrees) in comparison to the sun's rot axis. NB. if you use AIA, it's not flipped
			#also to deal with the flip that fits files have, you need to flip it upside down
			image_data = np.flip(image_data,0) #to deal with the fact it's a fits file
			mag_data = np.rot90(mag_data,2)
			mag_data = np.flip(mag_data,0)
			mag_data = abs(mag_data)

		self.R_sol = 6.597e8
		self.const_c = 3e8
		self.nt = Time.now()
		return image_data, mag_data

	def findcutoffs(self,arr_in,check_var):
		cut_off = 0
		count = 0
		nan_lines = []
		for i in np.arange(len(arr_in)):
			if i != count:
				cut_off = i
				break
			res = all(str(ele) == str(check_var) for ele in arr_in[i]) #checks if the whole line is nans
			if res==True: #if it is all nans, it increases the cut-off margin to include that line
				nan_lines.append(i)
				count+=1
		# print("count for nan lines", count)
		return cut_off
	
	def GetMuPhi(self,x,y, arr): #arr should be new_data

		r_pix = (len(arr)+len(arr[0]))/4
		x_dash = x-r_pix
		y_dash = r_pix-y
		r = np.sqrt((abs((x_dash)))**2+(abs((y_dash)))**2)
		# mu = np.cos(r*np.pi/(2*r_pix))
		mu = np.sqrt(1-abs((r/r_pix))**2)

		theta = math.atan(abs(y_dash)/abs(x_dash))
		phi = None
		if x_dash>=0 and y_dash>=0:
			phi = np.pi/2 - theta
		if x_dash>=0 and y_dash<=0:
			phi = np.pi/2 + theta
		if x_dash<=0 and y_dash<=0:
			phi = 2*np.pi - (theta+np.pi/2)
		if x_dash<=0 and y_dash>=0:
			phi = 2*np.pi - (np.pi/2 - theta)
		phi = 2*np.pi - phi
		#returns phi calculated from solar North anticlockwise, as pyobs also measures anticlockwise
		return mu, phi
	
	def GetMuPhi2D(self,arr): #arr should be new_data
		# print("shape of data ", np.shape(arr), len(arr[0][0])+len(arr[0][1]))
		r_pix = (len(arr[0][0])+len(arr[0]))/4 #the second one was arr[0][1]
		# print("r_pix", r_pix)
		x_dash = np.array(arr[0])-r_pix #change back to what it was
		# print("x_dash", x_dash)
		y_dash = r_pix-np.array(arr[1]) #this isnt right - arr[:,1]?
		# print("y_dash", y_dash)
		r_grid = np.sqrt((abs((x_dash)))**2+(abs((y_dash)))**2) #careful of r_pix = 0
		# mu = np.cos(r*np.pi/(2*r_pix))
		mu_grid = np.sqrt(1-(abs(r_grid/r_pix))**2)

		thetagrid = np.arctan2(abs(y_dash),abs(x_dash))
		phigrid = thetagrid
		# plt.imshow(phigrid)
		# plt.show()
		lenx = len(thetagrid[0])+1
		leny = len(thetagrid)+1
		halflenx = int(lenx/2)
		halfleny = int(leny/2)
		# if x_dash>=0 and y_dash>=0:
		phigrid[halflenx:lenx][halfleny:leny] = np.pi/2 - thetagrid[halflenx:lenx][halfleny:leny]
		# if x_dash>=0 and y_dash<=0:
		phigrid[0:halflenx][halfleny:leny] = np.pi/2 + thetagrid[0:halflenx][halfleny:leny]
		# if x_dash<=0 and y_dash<=0:
		phigrid[0:halflenx][0:halfleny] = 2*np.pi - (thetagrid[0:halflenx][0:halfleny]+np.pi/2)
		print(phigrid[0][halfleny-1])
		# if x_dash<=0 and y_dash>=0:
		phigrid[halflenx:lenx][0:halfleny] = 2*np.pi - (np.pi/2 - thetagrid[halflenx:lenx][0:leny])
		phigrid = 2*np.pi - phigrid
		# #returns phi calculated from solar North anticlockwise, as pyobs also measures anticlockwise

		return mu_grid, phigrid

	def find_min_idx(self,arr):
		k = arr.argmin() #np.nanargmin(arr)
		ncol = arr.shape[1]
		return k/ncol, k%ncol #row then column

	def find_mu_min(self,muarr,phiarr,limit):
		mus = np.array([0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,1.00]) #hard-coded
		# muarr = np.array([0.1,0.3,0.4])
		# mus = np.array([0.1,0.5,0.9])
		res = abs(muarr[None, :] - mus[:, None])
		# print("res:")
		# print(res)
		mins = np.min(res, axis=0)
		print("mins:", mins)
		valid_idx = np.where(mins <= limit)
		print("indices below the limit?", valid_idx)
		# out = valid_idx[mins[valid_idx].argmin()]
		# print(out)
		mu_points = muarr[valid_idx]
		phi_points = phiarr[valid_idx]

		return mu_points, phi_points

	def SolveSinQuart(self,A,B,C,D):
		term0 = float(-4*A*C + B**2 + 4*C*D)
		term1 = np.sqrt(term0)
		term2 = -1*(B+term1)/(2*C)
		x = np.arcsin(np.sqrt(term2))+2*np.pi
		return x

	def GetAngVel_SnodUlrich(self,latitude):
		Acoeff = 14.71
		Bcoeff = -2.39
		Ccoeff = -1.78
		sinl = np.sin(latitude)
		ang_vel_deg_pday = Acoeff+Bcoeff*sinl**2+Ccoeff*sinl**4
		return ang_vel_deg_pday

	def GetPredictiveOffset(self,key_obj, mu_in, phi_in):
		date_obs = key_obj.DATE__OBS
		# print("times", date_obs, nt.fits)
		ot = Time(str(date_obs[0][0:-1]), format='fits')
		dt_days = (self.nt.jd-ot.jd) #in days
		# dt_days = 1
		# print("time offset", dt_days)
		lat = self.R_sol*np.sin(np.arccos(mu_in))*np.cos(phi_in) # the physical distances, not the angles
		lon = self.R_sol*np.sin(np.arccos(mu_in))*np.sin(phi_in)
		ang_vel = self.GetAngVel_SnodUlrich(lat)
		lon_offset_rad = ang_vel*dt_days*np.pi/180 #in radians
		
		#check if longitude with offset is greater than R_sol, meaning it's gone off the disk in that time
		new_lon_angle = np.arcsin(lon/self.R_sol)+lon_offset_rad
		lon_pred = np.sin(new_lon_angle)*self.R_sol
		lat_pred = lat
		l_hyp = np.sqrt((abs((lon_pred)))**2+(abs((lat_pred)))**2)
		mu_pred = np.cos(np.arcsin(l_hyp/self.R_sol))
		theta_pred = math.atan(abs(lat_pred)/abs(lon_pred))
		phi_pred = None
		if lon_pred>=0 and lat_pred>=0:
			phi_pred = np.pi/2 - theta_pred
		if lon_pred>=0 and lat_pred<=0:
			phi_pred = np.pi/2 + theta_pred
		if lon_pred<=0 and lat_pred<=0:
			phi_pred = 2*np.pi - (theta_pred+np.pi/2)
		if lon_pred<=0 and lat_pred>=0:
			phi_pred = 2*np.pi - (np.pi/2 - theta_pred)
		# print(phi_pred, lon_pred, lat_pred)
		# phi_pred = phi_pred
		return mu_pred, phi_pred

#cropping the excess in the image
	def CropNans(self, image_data, cutoffs_checkvar):
		image_data_trans = image_data.T
		cut_off_top = self.findcutoffs(image_data,cutoffs_checkvar) - 1
		cut_off_bottom = len(image_data) - self.findcutoffs(np.flipud(image_data),cutoffs_checkvar)
		cut_off_left = self.findcutoffs(image_data_trans,cutoffs_checkvar) - 1
		cut_off_right = len(image_data_trans) - self.findcutoffs(np.flipud(image_data_trans),cutoffs_checkvar)
		truncated_data = image_data[cut_off_top:cut_off_bottom]
		trans_trunc_data = truncated_data.T[cut_off_left:cut_off_right]
		new_data = trans_trunc_data.T
		return new_data
	
	def MeanNoNans(self, new_data):
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
		return mean

	def FindPoints(self, bit_data, arra_data, intensity_limit):
		#===Finding the centre (the darkest point) of each spot
		contours = measure.find_contours(bit_data, intensity_limit) #uses the 'marching squares' method
		#https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py

		x_cens = []
		y_cens = []
		reduced_contours = []
		for e, contour in enumerate(contours):
			# if statement removes very small isolated points and the sun-is-a-spot issue
			if (len(contour)>50 and len(contour)<1000): #these numbers are currently pretty much arbitrary - might be worth checking
				low  = int(min(contour[:,1]))
				upp  = int(max(contour[:,1]))
				left = int(min(contour[:,0]))
				right= int(max(contour[:,0]))

				indiv_sharp = arra_data[left:right,low:upp]
				blur_geom_med = gaussian_filter(indiv_sharp, 4)
				# if e<=10:
				# 	plt.imshow(blur_geom_med)
				# 	plt.show()
				#error management for if blur_geom_med is limbward:
				if (len(blur_geom_med[0])==0 or len(blur_geom_med)==0):
					continue

				y_cen_spot,x_cen_spot = self.find_min_idx(blur_geom_med)
				y_cen = y_cen_spot+left
				x_cen = x_cen_spot+low
				# if e<100:
				# 	plt.imshow(blur_geom_med)
				# 	plt.scatter(x_cen_spot,y_cen_spot,color="green")
				# 	plt.show()
				# print("centres", x_cen, y_cen)
				x_cens.append(x_cen)
				y_cens.append(y_cen)
				reduced_contours.append(contour)
		return x_cens, y_cens, reduced_contours
	
	def FindPointsMuPhi(self,x_cens, y_cens, new_data):
		mu_cons = []
		phi_cons = []
		for i in np.arange(len(x_cens)):
			if str(x_cens[i])!="nan" or str(y_cens[i])!="nan":
				mu_con, phi_con = self.GetMuPhi(x_cens[i], y_cens[i], new_data)
			else:
				mu_con = "nan"
				phi_con = "nan"
			mu_cons.append(mu_con)
			phi_cons.append(phi_con)
		return mu_cons, phi_cons
	
	def FindPredMuPhi(self,mu_cons, phi_cons):
		mu_preds = []
		phi_preds = []
		for i in np.arange(len(mu_cons)):
			if str(mu_cons[i])!="nan":
				mu_pred, phi_pred = self.GetPredictiveOffset(self.keys, mu_cons[i], phi_cons[i])
				m=mu_pred
				p=np.rad2deg(phi_pred)
				mu_preds.append(m)
				phi_preds.append(p)
		return mu_preds, phi_preds

	def onclick(self,event,new_data):
		m,p=self.GetMuPhi(event.xdata, event.ydata, new_data)
		print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f zdata=%f mu=%f phi=%f' %
			('double' if event.dblclick else 'single', event.button,
			event.x, event.y, event.xdata, event.ydata, new_data[int(event.xdata)][int(event.ydata)],m,np.rad2deg(p)))
		return

	def PlotStatic(self,x_plots, y_plots, mu_plots, new_data): #put in whichever should be plotted, pred or current-to-sdo.
		fig, ax = plt.subplots()
		ax.imshow(new_data, cmap=plt.cm.gray)
		for i in np.arange(len(x_plots)):
			if str(mu_plots[i])!="nan":
				ax.annotate("%i"%(i), (x_plots[i], y_plots[i]))
				ax.scatter(x_plots[i], y_plots[i], color="white", s=5)
			else:
				ax.annotate("off disk, mu undefined", (x_plots[i], y_plots[i]))
		plt.show()
		return

	def PlotDynamic(self, data_image):
		fig, ax = plt.subplots()
		ax.imshow(data_image)
		ax.grid(False)
		oc = self.onclick
		cid = fig.canvas.mpl_connect('button_press_event', oc(data_image))
		plt.show()
		return
	
	def PrintLog(self,mu_plots,phi_plots):
		for i in np.arange(len(mu_plots)):
			print("key, mu, phi ", i, mu_plots[i], phi_plots[i])
		return
	
	def Mask_GreaterThan(self,ArrIn, Mask_value):
		B_masked = ma.masked_greater(ArrIn, Mask_value,copy=True)
		return B_masked
	def Mask_SmallerThanEq(self,ArrIn, Mask_value):
		arr = ma.masked_less_equal(ArrIn, Mask_value,copy=True)
		return arr
	
obj = FindSharps(mode="Spot", Predict=True, Verbose=True,Dynamic=False)