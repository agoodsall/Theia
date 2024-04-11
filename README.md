# Theia

### Description
Theia is a script that provides real-time coordinates of sunspots in helioprojective radial coordinates (HPR).
Theia follows the thresholding algorithms in Haywood et al. 2016 and Yeo et al. 2013 to separate sunspots, plage and quiet sun.
Currently Theia only provides coordinates for sunspots, not plage or faculae.
In order to do this, it queries the SDO database for the near-real-time data continuum data (Ic_noLimbDark_720s_nrt) and uses the intensity limit from Yeo et al. 2013 to find potential sunspots.

Theia is currently mainly used to produce a plot of potential targets with a key giving their coordinates in HPR.
For more information about the coordinate system, see Thompson 2006 (doi = {10.1051/0004-6361:20054262}).

Set-up
This code provides coordinates that should be fed to PyObs. For an FTS measurement, the mirror in the lab should be centred on the lower fibrehole, with a silver fibre. For VSS measurements, the mirror in the lab should be centred on the upper fibrehole, which has a yellow fibre.
Querying JSOC for SDO data requires a registered email address - this is goodsalljsocexport@gmail.com.

Future changes will include;
* An automatic version of Theia that calculates changes in coordinates and feeds a list of coordinates of the same spot to a file to be used by telescope software.
* A version of Theia that can provide coordinates for faculae and flares as well as sunspots.


### Requirements
* drms (http://jsoc.stanford.edu/jsocwiki/DRMSSetup, or via Sunpy)
* numpy 
* scipy
* astropy
* math
* skimage

### Inputs

* Predict, default True
The SDO data used is usually 2 hours behind our current time, so Theia has a predictive mode that uses differential rotation coefficients (see Spot, below) to determine how far in longitude the spot should have moved in the two hours.
Spots also have a latitude drift, but as this is a much smaller effect, it is not taken into consideration here.

Predict should be False if you are using Theia to find coordinates for archival data.
Predict should be True if you want current solar coordinates for in-realtime observations.

* Spot, default True
When 'Predict' is True, Theia uses certain differential rotation coefficients from Snodgrass and Ulrich 1990. The two options for coefficients describe either the rotation in doppler velocity space or the rotation of magnetic features.
If Spot is True, the Magnetic feature rotation coefficients are used.
If Spot is False, the Doppler feature rotation coefficients are used.

I recommend using the magnetic feature rotation coefficients, unless you are trying to find the coordinates to compare to quiet sun data taken before 2024.

* Verbose, default True
When Verbose is True, Theia prints the coordinates in mu and phi in the terminal, in relation to the key on the graph, e.g. :
Key Mu 	Phi
1	0.9	270.00

* Dynamic, default False


### Known Errors