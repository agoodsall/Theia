#code to produce an image giving the mu-psi coordinates for sharps using the near-real-time data from SDO

#notes
#cannot be used while using a vpn

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
plt.style.use(astropy_mpl_style)
plt.rcParams["figure.autolayout"] = True


class FindSharps():
	def __init__(self, Predict=True, Verbose=True, Dynamic=False, Spot=True):

		IntensityLimit = 0.89
		self.Spot = Spot
		# if Predict==False:
		# 	Time_0 = "2024.0"+str(values[0])+"."+str(values[1])+"0_09:00:00_TAI"
		# if Predict==True:
		# 	Time_0 = "2024.0"+str(values[0])+"."+str(values[1])+"0_07:00:00_TAI"
		# 	self.nt = Time("2024-0"+str(values[0])+"-"+str(values[1])+"0T07:00:00.000", format='fits') +1/12 #should be taken as units of days, so + 2 hours.
		# Time_0 = "$"
		data = self.Instantiate()
		cropdata = self.CropNans(data)
		centres_x, centres_y = self.FindPoints(cropdata, IntensityLimit)
		mus, phis = self.FindPointsMuPhi(centres_x,centres_y,cropdata)
		if Predict==True:
			mus, phis = self.FindPredMuPhi(mus,phis)
		else:
			phis=np.rad2deg(phis)
		if Verbose==True:
			self.PrintLog(mus, phis)
		if Dynamic==False:
			self.PlotStatic(centres_x, centres_y,mus,cropdata)
		if Dynamic==True:
			self.PlotDynamic(cropdata)
		

	def Instantiate(self,match_SDO=True, obs_datime = "$"):
		#===Instantiate
		c = drms.Client(email = "goodsalljsocexport@gmail.com",verbose=False)
		time_obs = obs_datime
		print(time_obs)
		sm = c.query('hmi.Ic_noLimbDark_720s_nrt['+time_obs+']', seg='Continuum') #this part isn't case sensitive
		self.keys = c.query('hmi.Ic_noLimbDark_720s_nrt['+time_obs+']', key=('DATE__OBS','CROTA2')) #CROTA2
		url_m = 'http://jsoc.stanford.edu' + sm.Continuum[0] #this part IS case sensitive
		print("data retrieved from:", url_m)
		image_data = fits.getdata(url_m)
		if match_SDO==True:
			image_data = np.rot90(image_data,2) #SDO HMI is upside down (like 180.02 degrees) in comparison to the sun's rot axis. NB. if you use AIA, it's not flipped
			#also to deal with the flip that fits files have, you need to flip it upside down
			image_data = np.flip(image_data,0)

		self.R_sol = 6.597e8
		self.const_c = 3e8
		self.nt = Time.now()
		# self.nt = Time("2024-03-10T07:00:00.000", format='fits') +1/12 #should be taken as units of days, so + 2 hours.
		# print(self.nt)
		return image_data

	def findcutoffs(self,arr_in):
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
	
	def GetMuPhi(self,x,y, arr): #arr should be new_data

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
			phi = 2*np.pi - (np.pi/2 - theta)
		# phi = 2*np.pi - phi
		#returns phi calculated from solar North anticlockwise, as pyobs also measures anticlockwise
		return mu, phi

	def find_min_idx(self,arr):
		k = arr.argmin() 
		ncol = arr.shape[1]
		return k/ncol, k%ncol #row then column

	def SolveSinQuart(self,A,B,C,D):
		term0 = float(-4*A*C + B**2 + 4*C*D)
		term1 = np.sqrt(term0)
		term2 = -1*(B+term1)/(2*C)
		x = np.arcsin(np.sqrt(term2))+2*np.pi
		return x

	def GetAngVel_SnodUlrich(self,latitude):
		if self.Spot==False:
			Acoeff = 14.71
			Bcoeff = -2.39
			Ccoeff = -1.78
		if self.Spot==True:
			#Below are the 1990 snodgrass papers' MAGNETIC rotation rate coefficients:
			Acoeff = 14.366
			Bcoeff = -2.297
			Ccoeff = -1.624
		sinl = np.sin(latitude)
		ang_vel_deg_pday = Acoeff+Bcoeff*sinl**2+Ccoeff*sinl**4
		return ang_vel_deg_pday

	def GetPredictiveOffset(self,key_obj, mu_in, phi_in):
		date_obs = key_obj.DATE__OBS
		# print("times", date_obs, nt.fits)
		ot = Time(str(date_obs[0][0:-1]), format='fits')
		dt_days = (self.nt.jd-ot.jd) #in days
		# dt_days = 1
		print("time offset", dt_days)
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
	def CropNans(self, image_data):
		image_data_trans = image_data.T
		cut_off_top = self.findcutoffs(image_data) - 1
		cut_off_bottom = len(image_data) - self.findcutoffs(np.flipud(image_data))
		cut_off_left = self.findcutoffs(image_data_trans) - 1
		cut_off_right = len(image_data_trans) - self.findcutoffs(np.flipud(image_data_trans))
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

	def FindPoints(self, new_data, intensity_limit):
		mean = self.MeanNoNans(new_data)
		#===Finding the centre (the darkest point) of each spot
		contours = measure.find_contours(new_data, intensity_limit*mean) #uses the 'marching squares' method
		#https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py

		x_cens = []
		y_cens = []

		for e, contour in enumerate(contours):
			# if statement removes very small isolated points and the sun-is-a-spot issue
			if len(contour)>50 and len(contour)<2000: #limits were 50 and 2000 pixels

				low  = int(min(contour[:,1]))
				upp  = int(max(contour[:,1]))
				left = int(min(contour[:,0]))
				right= int(max(contour[:,0]))

				indiv_sharp = new_data[left:right,low:upp]
				blur_geom_med = gaussian_filter(indiv_sharp, 4)

				#add in error management here if blur_geom_med is all nans or empty. - think this is when it's limbward
				if len(blur_geom_med)==0:
					print("gauss blur arr is empty")
					continue

				y_cen_spot,x_cen_spot = self.find_min_idx(blur_geom_med)

				y_cen = y_cen_spot+left
				x_cen = x_cen_spot+low
				# print("centres", x_cen, y_cen)
				x_cens.append(x_cen)
				y_cens.append(y_cen)
		return x_cens, y_cens
	
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
			if str(mu_cons[i])=="nan":
				mu_pred, phi_pred = self.GetPredictiveOffset(self.keys, mu_cons[i], phi_cons[i])
				mu_preds.append("nan")
				phi_preds.append("nan")
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
				ax.scatter(x_plots, y_plots, color="white", s=5)
			else:
				ax.annotate("off disk, mu undefined", (x_plots[i], y_plots[i]))
		plt.show()
		return

	def PlotDynamic(self, new_data):
		fig, ax = plt.subplots()
		ax.imshow(new_data)
		ax.grid(False)
		oc = self.onclick
		cid = fig.canvas.mpl_connect('button_press_event', oc(new_data))
		plt.show()
		return
	
	def PrintLog(self,mu_plots,phi_plots):
		for i in np.arange(len(mu_plots)):
			print("key, mu, phi ", i, mu_plots[i], phi_plots[i])
		return


obj = FindSharps()


# list = [(4,2)]
# for i in list:
# 	boole = True
# 	print("bool:", boole)
# 	print(i)
# 	objTheia = FindSharps(i,Predict=boole, Verbose=True,Dynamic=False) #predicted
# 	boole = False
# 	print("bool:", boole)
# 	objTheia = FindSharps(i,Predict=boole, Verbose=True,Dynamic=False) #true one
	