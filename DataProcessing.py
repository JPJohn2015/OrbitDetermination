'''
DataProcessing Library has functions for reading data files
provided by Lockheed Martin into dataframes of single orbital
passes.

Authors:    James Johnson

Created:    Feb-2020
'''

#Initialize OREkit and Java Virtual Machine
import orekit
vm = orekit.initVM()

#Import Orekit Libraries
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation.analytical.tle import TLE, SGP4
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngle, KeplerianOrbit
from org.orekit.attitudes import NadirPointing
from orekit.pyhelpers import *

#Import Python Libraries
import pandas as pd
import csv
import json
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta, timezone
from julian import from_jd
from math import *
import numpy as np
from numpy.linalg import norm
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from datetime import datetime, timedelta, timezone

#Orbit Determination Libraries
from Variables import *
from Transformations import *
from OrbitDetermination import *

########## FUNCTION LIST

def relativePath(relPath: str):
	'''Creates relative path from current file

    Attributes:
        relPath - path to file relative to this script

    Returns:
    	filePath - absolute file path generated
    '''
    #Figure out relative path
	scriptPath = os.path.dirname(os.path.abspath(__file__))
	filePath = os.path.join(scriptPath, relPath)

	return filePath

def createFile(noradID: int, detectionID: int, observations):
	'''Creates a file for a new NORAD ID/DetectionID that
	does not exist.

    Attributes:
        noradID - NORAD ID of object
        detectionID - Detection Request ID of object pass
        observations - Pandas Dataframe
    '''
    #Create a new file for the NORAD ID and Detection Request ID
	observations.to_pickle("measurement-objects/n" + str(noradID) + "d" + str(detectionID) + ".pkl")

def appendFile(noradID: int, detectionID: int, observations):
	'''Appends an already created file for a NORAD ID/DetectionID 
	with additional data. 

    Attributes:
        noradID - NORAD ID of object
        detectionID - Detection Request ID of object pass
        observations - Pandas Dataframe
    '''
    #Load pass object for the NORAD ID and Detection Request ID
	df = pd.read_pickle("measurement-objects/n" + str(noradID) + "d" + str(detectionID) + ".pkl")

	#Append new observations to file
	observations_new = df.append(observations)

	#Write new file
	observations_new.to_pickle("measurement-objects/n" + str(noradID) + "d" + str(detectionID) + ".pkl")

def readFile(noradID: int, detectionID: int):
	'''Reads a file for a NORAD ID/DetectionID 

    Attributes:
        noradID - NORAD ID of object
        detectionID - Detection Request ID of object pass
	
	Returns:
		dataframe - Pandas Dataframe
    '''
    #Load pass object for the NORAD ID and Detection Request ID
	dataframe = pd.read_pickle("measurement-objects/n" + str(noradID) + "d" + str(detectionID) + ".pkl")

	return dataframe

def newFilesToLoad():
	'''Looks in lockheed data directory and determines which
	files need to be loaded into pass objects

    Returns:
        newFileList - list of Excel File names
    '''
	#Load Files in the lockheed-data folder
	fileDirectory = [f for f in listdir("lockheed-data/") if isfile(join("lockheed-data/", f))]

	#Load the files processed dataframe
	if os.path.exists("measurement-objects/filesProcessed.xlsx") == False:
		#File does not exist, create file with currently processed files
		filesProcessed = []

		#Temporary Dataframe for files already processed
		df_temp = pd.DataFrame(filesProcessed,columns=['fileName'])
		df_temp.to_excel("measurement-objects/filesProcessed.xlsx")
	else:
		#No files already processed
		df_temp = pd.read_excel(relativePath("measurement-objects/filesProcessed.xlsx"))
		filesProcessed = df_temp['fileName'].tolist()

	#Initialize new file list
	newFileList = []

	#For each File in directory, check if already processed
	for i in range(0,len(fileDirectory)):
		if fileDirectory[i] not in filesProcessed:
			if fileDirectory[i] != '.DS_Store':
				#Add File to list to be processed
				newFileList.append(fileDirectory[i])
				filesProcessed.append(fileDirectory[i])

	#Update List of processed files
	updatedFiles = pd.DataFrame(filesProcessed,columns=['fileName'])
	updatedFiles.to_excel("measurement-objects/filesProcessed.xlsx")

	return newFileList

def initializeDataframeObjects():
	'''Takes in a list of Excel Files in the lockheed-data 
	folder and creates a dataframe file for each Norad ID and 
	Detection Request ID.

    Attributes:
        files - list of Excel File names
    '''
    #Determine which new files need to be loaded
	newFileList = newFilesToLoad()

	#If List is not empty, load new files
	if newFileList != []:	
		#Loading Data into Pandas Dataframe
		for i in range(0,len(newFileList)):
			if i == 0:
				#Create Dataframe for first file
				df = pd.read_excel(relativePath("lockheed-data/" + newFileList[i]))
			else:
				#Append Dataframe for subsequent files
				df_new = df.append(pd.read_excel(relativePath("lockheed-data/" + newFileList[i])))
				df = df_new

		#Drop columns of data that are not being used
		#LM_Observations Formatting
		df.drop(['Track Id','Total Detections','Num Tracks','Number of Records',
				 'Num Pixels','Frame Num','Flux Err','Flux','A','B','Row','Col'], axis=1)
		#CPObs Formatting
		#df.drop(['A','Airmass','B','Background Thresh', 'Camera Id', 'Classification Id', 'Col',
		#		 'Exp Time Sec','Flux','Flux Err','Frame Num','Geom El Deg','Instr Mag','Num Pixels',
		#		 'Num Tracks','Photo Mag', 'Photo Mag Err','Raw Light Peak','Row','Theta','Track Id'], axis=1)

		#List of NORAD ID's
		noradID = list(df['Norad Id'].unique())

		#Initialize Pass Dictionary
		passes = []

		#For each NORAD ID, load all Detection Request IDs
		for i in range(0,len(noradID)):
			#New Dataframe that matches NORAD ID
			df_norad = df[(df['Norad Id'] == noradID[i])]
			
			#List of Detection Request ID's for each NORAD ID
			detectionRequestID = list(df_norad['Detection Request Id'].unique())

			#For Each Detection Request ID, extract data for pass
			for j in range(0,len(detectionRequestID)):
				#New Dataframe that matches NORAD ID and Detection Request ID
				df_detection = df_norad[(df_norad['Detection Request Id'] == detectionRequestID[j])]
				
				#Determine Initial Julian Date for Pass Logs
				julianDate = float(df_detection['Time Jd Frac'].iloc[0]) + float(df_detection['Time Jd Int'].iloc[0])

				#Determine Pass Length for Pass Logs
				timeInitial = float(df_detection['Time Jd Frac'].iloc[0]) + float(df_detection['Time Jd Int'].iloc[0])
				timeFinal = float(df_detection['Time Jd Frac'].iloc[-1]) + float(df_detection['Time Jd Int'].iloc[-1])
				passLength = (timeFinal-timeInitial)*DAY

				#Check if File already Exists
				if os.path.exists("measurement-objects/n"+str(noradID[i])+"d"+str(detectionRequestID[j])+".pkl") == False:
					createFile(int(noradID[i]),int(detectionRequestID[j]),df_detection)
				
				#Append Pass Log
				passes.append([noradID[i], detectionRequestID[j], julianDate, passLength])

		#Log of Norad ID and Detection Request ID combinations
		df_passes = pd.DataFrame(passes,columns=['noradID','detectionRequestID','julianDate','passLength'])	
		df_passes.to_excel("measurement-objects/logOfPasses.xlsx")

def loadPassLog():
	'''Returns the pass log with Norad and Detection Request ID's
    '''
    #Load Pass Log file to dataframe
	passLog = pd.read_excel("measurement-objects/logOfPasses.xlsx")

	return passLog

def cubeSatProperties(size):
	'''Returns the phyisical properties of different sized CubeSats

    Attributes:
        size - string in the form of "nU" denoting CubeSat size

    Returns:
    	mass - mass of satellite in kg
    	area - cross sectional area of largest face in m^2
    	cd - coefficient of drag
    '''
    #Determine CubeSat Size
	if type(size) == int:
		cubeSatSize = size
	elif type(size) == str:
		cubeSatSize = int(size[0])
	else:
		cubeSatSize = size
    
    #Calculate Phyisical Properties of CubeSat
	mass = 1.33*cubeSatSize
	area = (0.01**3)*cubeSatSize

	#Estimates for Coefficients
	cd = 2.2
	cr = 1.5

	return mass, area, cd, cr

def statusUpdate(text):
	'''Returns a formatted terminal line with runtime and message

    Attributes:
        text - message to be printed in the terminal
        startTime - datetime object of start time of script
    '''
	#Calculate Runtime
	currentTime = datetime.now()
	runTime = currentTime - START_TIME

	#Calculates number of minutes and seconds
	m = floor(runTime.total_seconds()/60)
	s = floor(runTime.total_seconds() - m*60)
	ms = round((runTime.total_seconds() - m*60 - s)*100)

	#Format Milliseconds to two digits
	if ms == 100:
		ms = 0

	#Format String with time
	timeString = '['+'{:0>2}'.format(m)+':'+'{:0>2}'.format(s)+'.'+'{:0>2}'.format(ms)+'] | '

	#Print formatted String
	if type(text) == str:
		print(timeString + text)
	else:
		print(timeString + str(text))

def userInput(prompt):
	'''Returns a formatted terminal line with runtime and message

    Attributes:
        text - message to be printed in the terminal
        startTime - datetime object of start time of script
    '''
	#Calculate Runtime
	currentTime = datetime.now()
	runTime = currentTime - START_TIME

	#Calculates number of minutes and seconds
	m = floor(runTime.total_seconds()/60)
	s = floor(runTime.total_seconds() - m*60)
	ms = round((runTime.total_seconds() - m*60 - s)*100)

	#Format String with time
	timeString = '['+'{:0>2}'.format(m)+':'+'{:0>2}'.format(s)+'.'+'{:0>2}'.format(ms)+'] | '

	#Print formatted String
	response = input(timeString + prompt)

	return response

def getTLE(noradID,searchDate):
	'''Get the nearest TLE to the specified time for the NORAD ID

    Attributes:
        noradID - NORAD ID of object
        searchDate - datetime of IOD Epoch

    Returns:
    	tle - orekit TLE object
    '''
	#Initialize SpaceTrack API Client
	st = SpaceTrackClient(identity=USERNAME, password=PASSWORD)
	rawTle = st.tle(norad_cat_id=noradID, epoch='<{}'.format(searchDate), 
					orderby='epoch desc', limit=1, format='tle')

	#TLE Lines
	tleLine1 = rawTle.split('\n')[0]
	tleLine2 = rawTle.split('\n')[1]

	print(tleLine1)
	print(tleLine2)

	#TLE Object
	tle = TLE(tleLine1, tleLine2)

	return tle


########## CLASS LIST

class Pass:

	def __init__(self, noradID, detectionID):
		'''Initializes the Pass class

		Attributes:
			noradID - NORAD ID of object
			detectionID - Detection Request ID of object pass
		'''
		#Define Station Location
		self.latitude = LATITUDE
		self.longitude = LONGITUDE
		self.altitude = ALTITUDE

		#Define ID numbers
		self.noradID = noradID
		self.detectionID = detectionID

		#Reading Dataframe file
		df = readFile(noradID, detectionID)

		#Define Measurement Lists
		self.azimuth = list(df["Az Deg"])
		self.elevation = list(df["Obs El Deg"])
		self.rightAscension = list(df["ra J2000 deg"])
		self.declination = list(df["dec J2000 deg"])
		self.julianDateFrac = list(df["Time Jd Frac"])
		self.julianDateInt = list(df["Time Jd Int"])

		#Calculated Values
		self.datetime = self.julianDateToDatetime()
		self.slantRange = self.slantRangeVector()
		self.rSite = self.stationSite()
		self.initialEpoch = self.datetime[0]
		self.middleEpoch = self.datetime[int(len(self.datetime)/2)]
		self.finalEpoch = self.datetime[-1]

	def julianDateToDatetime(self):
		'''Converts a List of Julian Dates into
		a list of Datetime objects.

		Returns:
			datetime - datetime object
		'''
		datetime = []
		for i in range(0,len(self.julianDateFrac)):
			datetime.append(from_jd(self.julianDateInt[i] + self.julianDateFrac[i]))

		return datetime

	def slantRangeVector(self):
		'''Calculates a list of Slant Range vectors from
		list of Right Ascensnion and Declination angles.

		Returns:
			slantRangeVector - slant range unit vector
		'''
		slantRangeVector = []
		for i in range(0,len(self.rightAscension)):
			#Convert Angle Measurements to radians
			ra = radians(self.rightAscension[i])
			dec = radians(self.declination[i])

			#Calculate slant range vector
			slantRange = Vector3D(float(cos(dec)*cos(ra)),
                           float(cos(dec)*sin(ra)),
                           float(sin(dec)))
			slantRangeVector.append(slantRange)

		return slantRangeVector

	def stationSite(self):
		'''Calculates the site vector for the ground station in ECI
		at the desired time from the latitude, longitude and altitude 

		Returns:
			rSiteList - List of Vector3D of the ground station site vector
		'''
		#Define Earth Parameters as easier variables
		Re = EARTH_RADIUS
		f = EARTH_FLATTENING

		rSiteList = []
		for i in range(0,len(self.datetime)):
			#Calculate LST
			lst = localSiderealTime(self.datetime[i])

			#Calculate Vector Coefficients
			C1 = ((Re/sqrt(1-(2*f-(f**2))*(sin(radians(LATITUDE))**2)))+(ALTITUDE/1000))*cos(radians(LATITUDE))
			C2 = (((Re*(1-f)**2)/sqrt(1-(2*f-(f**2))*(sin(radians(LATITUDE))**2)))+(ALTITUDE/1000))

			#Final Size Vector
			rSite = Vector3D(C1*cos(radians(lst)),C1*sin(radians(lst)),C2*sin(radians(LATITUDE)))
			rSiteList.append(rSite)

		return rSiteList
	
	def getPassLength(self):
		'''Calculates the time interval for the measurements.

		Returns:
			passLength - length of satellite pass
		'''
		#Initial and Final Times
		timeInitial = float(self.julianDateInt[0]) + float(self.julianDateFrac[0])
		timeFinal = float(self.julianDateInt[-1]) + float(self.julianDateFrac[-1])

		return (timeFinal-timeInitial)*24*60*60

	def getNumberOfPoints(self):
		'''Returns the number of data points in the pass.

		Returns:
			numberOfPoints - number of data points
		'''
		return len(self.azimuth)

	def getAverageStepSize(self):
		'''Returns the average step size between data points in the pass.

		Returns:
			averageStepSize - average spacing between data points
		'''	
		#List of stepsizes between data points
		stepsize = []
		for i in range(1,len(self.datetime)):
			stepsize.append(self.datetime[i] - self.datetime[i-1])

		return np.mean(stepsize).total_seconds()

	def getConstantSpacing(self, stepSize):
		'''Returns an object with specified spacing and point parameters
		
		Attributes:
        	stepSize - desired time spacing between measurements
	
		Returns:
			azimuth - modified list of Azimuth angles
			elevation - modified list of Elevation angles
			rightAscension - modified list of Right Ascension angles
			declination - modified list of Declination angle
			datetime - modified list of Datetime objects
			slantRange - modified list of Slant Rane Vectors
		'''
		#Spacing between data points to get correct stepsize
		averageStepSize = self.getAverageStepSize()
		spacing = int(stepSize/averageStepSize)

		#Define data lists
		azimuth = []
		elevation = []
		rightAscension = []
		declination = []
		datetime = []
		slantRange = []
		rSite = []
		for i in range(0,len(self.azimuth),spacing):

			#Pull data points that match the proper spacing
			azimuth.append(self.azimuth[i])
			elevation.append(self.elevation[i])
			rightAscension.append(self.rightAscension[i])
			declination.append(self.declination[i])
			datetime.append(self.datetime[i])
			slantRange.append(self.slantRange[i])
			rSite.append(self.rSite[i])

		return azimuth, elevation, rightAscension, declination, datetime, slantRange, rSite

	def getThreePointCentered(self,stepSize):
		'''Returns an object with specified spacing and point parameters
		
		Attributes:
        	stepSize - desired time spacing between measurements
	
		Returns:
			azimuth - modified list of Azimuth angles
			elevation - modified list of Elevation angles
			rightAscension - modified list of Right Ascension angles
			declination - modified list of Declination angle
			datetime - modified list of Datetime objects
			slantRange - modified list of Slant Range Vectors
			rSite - modified list of Site Vectors
		'''
		#Orbital Pass Parameters 
		averageStepSize = self.getAverageStepSize()	
		totalLength = len(self.azimuth)

		#Get indexes for three points centered on pass
		valid = False
		while valid == False:
			#Calculate index of Center, Left and Right Points
			idxCenter = int(totalLength/2)
			idxLeft = int(idxCenter - (stepSize/averageStepSize))
			idxRight = int(idxCenter + (stepSize/averageStepSize))

			#Check if values are outside of possible range and correct stepsize	
			if idxLeft <= 1 or idxRight >= totalLength-1:
				stepSize = stepSize - 1
				valid = False
			else:
				valid = True

		#Define data lists
		azimuth = [self.azimuth[idxLeft],self.azimuth[idxCenter],self.azimuth[idxRight]]
		elevation = [self.elevation[idxLeft],self.elevation[idxCenter],self.elevation[idxRight]]
		rightAscension = [self.rightAscension[idxLeft],self.rightAscension[idxCenter],self.rightAscension[idxRight]]
		declination = [self.declination[idxLeft],self.declination[idxCenter],self.declination[idxRight]]
		datetime = [self.datetime[idxLeft],self.datetime[idxCenter],self.datetime[idxRight]]
		slantRange = [self.slantRange[idxLeft],self.slantRange[idxCenter],self.slantRange[idxRight]]
		rSite = [self.rSite[idxLeft],self.rSite[idxCenter],self.rSite[idxRight]]

		return azimuth, elevation, rightAscension, declination, datetime, slantRange, rSite

	def threePointCenteredAngleRates(self,stepSize):
		'''Returns an object with specified spacing and point parameters
		of the angle rates
		
		Attributes:
        	stepSize - desired time spacing between measurements
	
		Returns:
			azimuthRate - modified list of Azimuth rates
			elevationRate - modified list of Elevation rates
			rightAscensionRate - modified list of Right Ascension rates
			declinationRate - modified list of Declination rates
		'''
		#Orbital Pass Parameters 
		averageStepSize = self.getAverageStepSize()	
		totalLength = len(self.azimuth)

		#Get indexes for three points centered on pass
		valid = False
		while valid == False:
			#Calculate index of Center, Left and Right Points
			idxCenter = int(totalLength/2)
			idxLeft = int(idxCenter - (stepSize/averageStepSize))
			idxRight = int(idxCenter + (stepSize/averageStepSize))

			#Check if values are outside of possible range and correct stepsize	
			if idxLeft <= 0 or idxRight >= totalLength:
				stepSize = stepSize - 1
				valid = False
			else:
				valid = True

		#Define data lists
		azimuthRate = [self.getAngleRate('az',idxLeft),self.getAngleRate('az',idxCenter),self.getAngleRate('az',idxRight)]
		elevationRate = [self.getAngleRate('el',idxLeft),self.getAngleRate('el',idxCenter),self.getAngleRate('el',idxRight)]
		rightAscensionRate = [self.getAngleRate('ra',idxLeft),self.getAngleRate('ra',idxCenter),self.getAngleRate('ra',idxRight)]
		declinationRate = [self.getAngleRate('dec',idxLeft),self.getAngleRate('dec',idxCenter),self.getAngleRate('dec',idxRight)]

		return azimuthRate, elevationRate, rightAscensionRate, declinationRate

	def getAngleRate(self,angle,index):
		'''Three Point Centered Finite Difference of Angle Rates
		
		Attributes:
        	angle - desired angle to finite difference
        	index - index to finite difference centered on
	
		Returns:
			angleRate - approximate angle rate
		'''
		#Determine step size of finite differencing
		stepSize = (self.datetime[index+1]-self.datetime[index-1]).total_seconds()

		#Determine which angle to approximate
		if angle == 'az':
			#Azimuth rate
			angleRate = (self.azimuth[index+1]-self.azimuth[index-1])/stepSize
		elif angle == 'el':
			#Elevation Rate
			angleRate = (self.elevation[index+1]-self.elevation[index-1])/stepSize
		elif angle == 'ra':
			#Right Ascension Rate
			angleRate = (self.rightAscension[index+1]-self.rightAscension[index-1])/stepSize
		elif angle == 'dec':
			#Declination Rate
			angleRate = (self.declination[index+1]-self.declination[index-1])/stepSize
		else:
			#Invalid Angle Designation
			error('Invalid Angle Designation')

		return angleRate

	def getFirstAndLast(self):
		'''Returns data lists with first and last data point in the pass
	
		Returns:
			azimuth - modified list of Azimuth angles
			elevation - modified list of Elevation angles
			rightAscension - modified list of Right Ascension angles
			declination - modified list of Declination angle
			datetime - modified list of Datetime objects
			slantRange - modified list of Slant Rane Vectors
		'''
		#Define data lists
		azimuth = [self.azimuth[0],self.azimuth[-1]]
		elevation = [self.elevation[0],self.elevation[-1]]
		rightAscension = [self.rightAscension[0],self.rightAscension[-1]]
		declination = [self.declination[0],self.declination[-1]]
		datetime = [self.datetime[0],self.datetime[-1]]
		slantRange = [self.slantRange[0],self.slantRange[-1]]
		rSite = [self.rSite[0],self.rSite[-1]]

		return azimuth, elevation, rightAscension, declination, datetime, slantRange, rSite


class GenerateIODReport:

	def __init__(self, InitialOrbitDetermination, TLELines):
		'''Initializes the IOD Report class
		'''
		#Define IOD And Orbit Parameters
		self.iod = InitialOrbitDetermination
		self.orbit = self.iod.initialOrbit
		self.method = self.iod.method
		self.epoch = self.orbit.getDate()
		self.TLELines = TLELines

		#Define Object Parameters
		self.noradID = self.iod.noradID
		self.detectionRequestID = self.iod.detectionRequestID

		#Define InitialOrbitDetermination Parameters
		self.passLength = self.iod.passLength
		self.arcLength = self.iod.alpha13
		self.scale = self.iod.scale

	def defineOrbitalElements(self,orbit):
		'''Defines the classical orbital elements from the orekit
		orbit class.
		'''
		#Semi-Major axis in km
		a = orbit.getA()*M2KM

		#Eccentricity
		e = orbit.getE()

		#Inclination
		i = degrees(orbit.getI())

		#Right Ascension of Ascending Node
		if orbit.getRightAscensionOfAscendingNode() < 0:
			r = 360 + degrees(orbit.getRightAscensionOfAscendingNode())
		else:
			r = degrees(orbit.getRightAscensionOfAscendingNode())

		#Argument of Perigee
		if orbit.getPerigeeArgument() < 0:
			w = 360 + degrees(orbit.getPerigeeArgument())
		else:
			w = degrees(orbit.getPerigeeArgument())

		#True Anomaly
		if orbit.getTrueAnomaly() < 0:
			v = 360 + degrees(orbit.getTrueAnomaly())
		else:
			v = degrees(orbit.getTrueAnomaly())

		#Period and Mean Motion
		T = 2*pi*np.sqrt(np.absolute(a**3)/EARTH_MU_KM)
		n = T/(2*pi)

		return [a, e, i, r, w, v, T, n]

	def defineCartesianVectors(self,orbit):
		'''Defines the cartesian state vectors based on the orekit orbit
		class.
		'''
		#Format orbit object to state vector
		pv = orbit.getPVCoordinates()
		state = PVCoordinatesToList(pv)

		#Define position and velocity components
		x = state[0]*M2KM
		y = state[1]*M2KM
		z = state[2]*M2KM
		vx = state[3]*M2KM
		vy = state[4]*M2KM
		vz = state[5]*M2KM

		return [x, y, z, vx, vy, vz]

	def defineTLEComparison(self):

		#Get TLE Data to compare
		self.tle = TLE(self.TLELines[0], self.TLELines[1])
		#self.tle = getTLE(self.noradID,self.orbit.getDate())

		propagator = createSGP4Propagator(self.tle,1.0,1.0,1.0,1.0,self.orbit.getDate())
		pv = propagator.propagate(self.orbit.getDate()).getPVCoordinates()
		self.tleOrbit = KeplerianOrbit(pv, ECI, WGS84ELLIPSOID.getGM())

		tleElements = self.defineOrbitalElements(self.tleOrbit)

		self.period = tleElements[6]

		return tleElements

	def textFileReportIOD(self):

		state = self.defineCartesianVectors(self.orbit)
		elements = self.defineOrbitalElements(self.orbit)
		oneOrbit, oneOrbitShifted = self.orbitError()

		string1 = '{:0>5d}'.format(self.noradID) +'  '+ '{:.2f}'.format(self.passLength) +'  '+ \
				  '{:.3f}'.format(self.arcLength) +'  '+ '{:.2f}'.format(self.scale) +'  '
		string2 = '{:.3f}'.format(elements[0]) +'  '+ '{:.5f}'.format(elements[1]) +'  '+ '{:.3f}'.format(elements[2]) +'  '+ \
				  '{:.3f}'.format(elements[3]) +'  '+ '{:.3f}'.format(elements[4]) +'  '+ '{:.3f}'.format(elements[5]) +'  '+ '{:.3f}'.format(elements[6]) +'  ' 
		string3 = '{:.3f}'.format(state[0]) +'  '+ '{:.3f}'.format(state[1]) +'  '+ '{:.3f}'.format(state[2]) +'  '+ \
				  '{:.6f}'.format(state[3]) +'  '+ '{:.6f}'.format(state[4]) +'  '+ '{:.6f}'.format(state[5]) +'  '
		string4 = '{:.3f}'.format(oneOrbit[0]) +'  '+ '{:.4f}'.format(oneOrbit[1]) +'  '+ '{:.3f}'.format(oneOrbit[2]) +'  '+ '{:.3f}'.format(oneOrbit[3]) +'  '
		string5 = '{:.3f}'.format(oneOrbitShifted[0]) +'  '+ '{:.3f}'.format(oneOrbitShifted[1]) +'  '+ '{:.3f}'.format(oneOrbitShifted[2])

		print(string1 + string2 + string3 + string4 + string5)

	def textFileReportTLE(self):

		self.defineTLEComparison()

		state = self.defineCartesianVectors(self.tleOrbit)
		elements = self.defineOrbitalElements(self.tleOrbit)
		oneOrbit, oneOrbitShifted = self.orbitError()

		string1 = '{:0>5d}'.format(self.noradID) +'  '+ '{:.2f}'.format(self.passLength) +'  '+ \
				  '{:.3f}'.format(self.arcLength) +'  '+ '{:.2f}'.format(self.scale) +'  '
		string2 = '{:.3f}'.format(elements[0]) +'  '+ '{:.5f}'.format(elements[1]) +'  '+ '{:.3f}'.format(elements[2]) +'  '+ \
				  '{:.3f}'.format(elements[3]) +'  '+ '{:.3f}'.format(elements[4]) +'  '+ '{:.3f}'.format(elements[5]) +'  '+ '{:.3f}'.format(elements[6]) +'  ' 
		string3 = '{:.3f}'.format(state[0]) +'  '+ '{:.3f}'.format(state[1]) +'  '+ '{:.3f}'.format(state[2]) +'  '+ \
				  '{:.6f}'.format(state[3]) +'  '+ '{:.6f}'.format(state[4]) +'  '+ '{:.6f}'.format(state[5]) +'  '
		string4 = '{:.3f}'.format(oneOrbit[0]) +'  '+ '{:.4f}'.format(oneOrbit[1]) +'  '+ '{:.3f}'.format(oneOrbit[2]) +'  '+ '{:.3f}'.format(oneOrbit[3]) +'  '
		string5 = '{:.3f}'.format(oneOrbitShifted[0]) +'  '+ '{:.3f}'.format(oneOrbitShifted[1]) +'  '+ '{:.3f}'.format(oneOrbitShifted[2])

		print(string1 + string2 + string3 + string4 + string5)

	def orbitError(self):

		#Initialize TLE comparison
		self.defineTLEComparison()

		#Set up propagators
		epoch = self.orbit.getDate()
		tlePropagator = createPropagator(self.tleOrbit,1.0,1.0,1.0,1.0)
		iodPropagator = createPropagator(self.orbit,1.0,1.0,1.0,1.0)

		iodPeriod = 2*pi*np.sqrt(((self.orbit.getA()*M2KM)**3)/EARTH_MU_KM)
		dPeriod = np.absolute(self.period - iodPeriod)
		
		#Check if perigee is below 150 km
		if self.orbit.getA() < 0:
			#No propagation error
			oneOrbit = [0.0, 0.0, 0.0, 0.0]
			oneOrbitShifted = [0.0, 0.0, 0.0]

		else:
			#One orbit propagation
			oneOrbitDate = epoch.shiftedBy(float(self.period))
			pvTLE = PVCoordinatesToList(tlePropagator.propagate(oneOrbitDate).getPVCoordinates())
			pvIOD = PVCoordinatesToList(iodPropagator.propagate(oneOrbitDate).getPVCoordinates())
			raTLE, decTLE = stateVectorToRaDec(pvTLE,absolutedate_to_datetime(oneOrbitDate))
			raIOD, decIOD = stateVectorToRaDec(pvIOD,absolutedate_to_datetime(oneOrbitDate))

			#One orbit error
			dX = np.subtract(pvTLE,pvIOD)
			dRa = raTLE-raIOD
			dDec = decTLE-decIOD
			dPos = np.linalg.norm(dX[0:3])*M2KM
			dVel = np.linalg.norm(dX[3:6])*M2KM

			oneOrbit = [dPos, dVel, dRa, dDec]

			#One orbit propagation shifted
			time = []
			raErr = []
			decErr = []
			angleErr = []
			posErr = []
			velErr = []
			for i in range(int(self.period - 2*dPeriod),int(self.period + 2*dPeriod),1):
				shifter = epoch.shiftedBy(float(i))
				pvIOD = PVCoordinatesToList(iodPropagator.propagate(shifter).getPVCoordinates())
				raIOD, decIOD = stateVectorToRaDec(pvIOD,absolutedate_to_datetime(shifter))
				dX = np.subtract(pvTLE,pvIOD)

				time.append(self.period - i)
				raErr.append(raTLE-raIOD)
				decErr.append(decTLE-decIOD)
				posErr.append(np.linalg.norm(dX[0:3])*M2KM)
				velErr.append(np.linalg.norm(dX[3:6])*M2KM)
				angleErr.append(np.sqrt((np.absolute(raTLE-raIOD))**2 + (np.absolute(decTLE-decIOD))**2))

			idx = angleErr.index(min(angleErr))

			dRa = raErr[idx]
			dDec = decErr[idx]
			dt = time[idx]

			oneOrbitShifted = [dRa, dDec,dt]

		return oneOrbit, oneOrbitShifted

	def oneOrbitError(self):

		self.defineTLEComparison()

		epoch = self.orbit.getDate()
		tlePropagator = createPropagator(self.tleOrbit,1.0,1.0,1.0,1.0)
		iodPropagator = createPropagator(self.orbit,1.0,1.0,1.0,1.0)

		estimationError = []
		timeError = []
		for i in range(0,int(self.period),MINUTE):
			pvTLE = PVCoordinatesToList(tlePropagator.propagate(epoch.shiftedBy(float(i))).getPVCoordinates())
			pvPOD = PVCoordinatesToList(iodPropagator.propagate(epoch.shiftedBy(float(i))).getPVCoordinates())

			dX = np.subtract(pvTLE,pvPOD)
			error = np.linalg.norm(dX[0:3])/KM2M

			estimationError.append(error)
			timeError.append(absolutedate_to_datetime(epoch.shiftedBy(float(i))))

		print(estimationError)

	def oneDayError(self):

		self.defineTLEComparison()

		epoch = self.orbit.getDate()
		tlePropagator = createPropagator(self.tleOrbit,1.0,1.0,1.0,1.0)
		iodPropagator = createPropagator(self.orbit,1.0,1.0,1.0,1.0)

		estimationError = []
		timeError = []
		for i in range(0,DAY,10*MINUTE):
			pvTLE = PVCoordinatesToList(tlePropagator.propagate(epoch.shiftedBy(float(i))).getPVCoordinates())
			pvPOD = PVCoordinatesToList(iodPropagator.propagate(epoch.shiftedBy(float(i))).getPVCoordinates())

			dX = np.subtract(pvTLE,pvPOD)
			error = np.linalg.norm(dX[0:3])/KM2M

			estimationError.append(error)
			timeError.append(absolutedate_to_datetime(epoch.shiftedBy(float(i))))

		print(estimationError)










