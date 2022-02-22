'''
Orbit Determination Library has functions for performing
Initial and Statistical Orbit Determination.

Authors:    James Johnson

Created:    Feb-2020
'''

#Initialize OREkit and Java Virtual Machine
import orekit
vm = orekit.initVM()
#Set up OREkit Directory
from orekit.pyhelpers import setup_orekit_curdir, JArray_double2D
setup_orekit_curdir()

#Import Orekit Libraries
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from org.orekit.frames import FramesFactory, ITRFVersion, TopocentricFrame, LocalOrbitalFrame, LOFType, Transform
from org.orekit.utils import IERSConventions, Constants, CartesianDerivativesFilter, PVCoordinates, TimeStampedPVCoordinates
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.propagation.analytical.tle import TLE, SGP4
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.attitudes import NadirPointing
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngle, KeplerianOrbit
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.models.earth.atmosphere import NRLMSISE00, SimpleExponentialAtmosphere
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.forces.gravity import Relativity
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.estimation.iod import IodGooding
from org.orekit.estimation.measurements import AngularAzEl, GroundStation, ObservableSatellite
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.linear import Array2DRowRealMatrix
from org.hipparchus.geometry.euclidean.threed import Vector3D, Rotation

from orekit.pyhelpers import *
from orekit import JArray, JArray_double

#Import Python Libraries
from java.io import File
from datetime import datetime, timedelta
from math import *
import numpy as np
import matplotlib.pyplot as plt
from julian import from_jd

#Import Orbit Determination Libraries
from DataProcessing import *
from Transformations import *
from OrbitDetermination import *
from Variables import *


class StatisticalOrbitDetermination:

	def __init__(self, noradID, detectionRequestID, initialOrbit):
		'''Initializes the StatisticalOrbitDetermination class

		Attributes:
			noradID - NORAD ID of object
			detectionRequestID - List of detection RequestID's to process
			initialOrbit - orekit orbit object
			numberOfPasses - number of passes to process
			measurementSpacing - time spacing (s) between measurements
		'''

		#Define Initializing Variables
		self.noradID = noradID
		self.detectionRequestID = detectionRequestID
		self.initialOrbit = initialOrbit

		#Measurement Weighting unless user defined
		self.measurementWeighting = [[1,0],[0,1]]

		#Object Properties unless user defined
		self.mass = 1
		self.area = 1
		self.cd = 2.2
		self.cr = 1.5

	def defineMeasurementWeighting(self, W):
		'''Defines the measurement weighting matrix

		Attributes:
			W - weighting matrix
		'''
		#Weighting Matrix
		self.measurementWeighting = W

	def defineObjectProperties(self,mass,area,cd,cr):
		'''Defines the object's physical properties

		Attributes:
			mass - mass of object
			area - frontal surface area of object
			cd - estimated coefficient of drag
			cr - estimated coefficient of relectivity
		'''
		#Define Object Properties
		self.mass = mass
		self.area = area
		self.cd = cd
		self.cr = cr

	def estimate(self,numberOfPasses,measurementSpacing):

		#Define Estimation Parameters
		self.numberOfPasses = numberOfPasses
		self.measurementSpacing = measurementSpacing

		#Set up Propagator
		initialEstimate = createPropagator(self.initialOrbit,self.mass,self.area,self.cd,self.cr)
		currentEstimate = createPropagator(self.initialOrbit,self.mass,self.area,self.cd,self.cr)

		#Initialize Tracking Variables
		self.trackRMS = []
		self.trackIter = []
		iteration = 0
		measurement = 0

		#Initialize  RMS values to enter loop
		RMSold = 0
		RMSnew = 1
		dRMS = RMSnew - RMSold

		#Number of Iterations of Batch Least Squares
		while np.linalg.norm(dRMS) >= TOLERANCE and iteration < MAX_ITERATIONS:
	    #for k in range(0,1):
			#Initialize Tracking Variables
			self.raGuess = []
			self.decGuess = []
			self.raMeasure = []
			self.decMeasure = []
			self.measurementNumber = []
			self.time = []

			#Initialize Matrix Sums
			sumAtWA = []
			sumAtWb = []
			sumb = []

			#For each Orbital Pass being processed
			for j in range(0,self.numberOfPasses):
				#Process Next Orbital Pass
				orbitalPass = Pass(self.noradID, self.detectionRequestID[j])
				az, el, ra, dec, datetime, slantRange, rSite = orbitalPass.getConstantSpacing(measurementSpacing)

				#For each measurement in th Orbital Pass
				for i in range(0,len(datetime)):

					#For each dimesion in the State Fector
					for z in range(0,6):
						#Modify the State for current Dimension
						currentEstimate, __ = modifyPropagator('clear',initialEstimate,self.mass,self.area,self.cd,self.cr)
						modifiedEstimate, dX = modifyPropagator(z,initialEstimate,self.mass,self.area,self.cd,self.cr)

						#Propagating State to measurement
						pvi = currentEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
						Xi = PVCoordinatesToList(pvi)

						#Modified State
						pvMod = modifiedEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
						Xmod = PVCoordinatesToList(pvMod)

						#Calculate Matricies
						A = self.calculateFiniteDifference(Xi,Xmod,dX,datetime[i])
						b = self.calculateResiduals(Xi,ra[i],dec[i],datetime[i]) 
						AtWA, AtWb, btWb = self.calculateCovarianceMatricies(A,b)

						#Sum Matricies for each measurement
						if self.time == []:
							sumAtWA = AtWA
							sumAtWb = AtWb
							sumbtWb = btWb
						else:
							sumAtWA = np.add(sumAtWA,AtWA)
							sumAtWb = np.add(sumAtWb,AtWb)
							sumbtWb = np.add(sumbtWb,btWb)

						#Reset Propagators to initial estimate
						currentEstimate = modifyPropagator('clear',initialEstimate,self.mass,self.area,self.cd,self.cr)
						modifiedEstimate = modifyPropagator('clear',initialEstimate,self.mass,self.area,self.cd,self.cr)

					#Right Ascension and Declination of current state
					rai, deci = stateVectorToRaDec(Xi,datetime[i])

					#Tracking Variables
					self.raMeasure.append(ra[i])
					self.decMeasure.append(dec[i])
					self.raGuess.append(rai)
					self.decGuess.append(deci)
					self.time.append(datetime[i])

			#Average the Root Mean Square of residuals
			btWbAvg = np.divide(sumbtWb,len(self.time))

			#Inverse of AtWA matrix
			u, s, v = np.linalg.svd(sumAtWA,full_matrices=True)

			#Calculate Covariance Matrix
			P = np.matmul(np.matmul(np.transpose(v),np.linalg.inv(np.diag(s))),np.transpose(u))
			eigval, eigvect = np.linalg.eig(P[0:3,0:3])

			#Calculate Root Mean Square for iteration
			RMS = sqrt(btWbAvg/(2*len(self.time)))
	        
			#Change in Initial State
			dX = np.matmul(P,sumAtWb)
			initialEstimate = updatePropagator(dX,initialEstimate,self.mass,self.area,self.cd,self.cr)

			#Compare the RMS with previous iteration for exit condition
			RMSold = RMSnew
			RMSnew = RMS
			dRMS = RMSnew - RMSold

			#Counter and Tracking Variables
			iteration += 1
			self.trackRMS.append(RMS)
			self.trackIter.append(iteration)

	        #Status Update
			if iteration == 1 or iteration == 5 or iteration == 10:
				yn = userInput('Change in RMS is currently '+format(np.absolute(dRMS), "5.1e")+' at '+str(iteration)+' iterations. Continue? (y/n): ')
				if yn == 'n':
					break
				elif yn == 'y':
					continue 
				else:
					error('invalid option')
			else:
				statusUpdate(iteration)

		self.measurementNumber = []
		for i in range(0,len(self.raGuess)):
			self.measurementNumber.append(i)

		self.finalOrbit = initialEstimate.getInitialState().getOrbit()
		self.finalEstimate = initialEstimate

		return self.finalOrbit

	def calculateFiniteDifference(self,Xi,Xmod,dX,datetime):
		'''Calculates the Jacobian Matrix of the observations
		with the current state vector

		Attributes:
			Xi - state vector at current timestep
			Xmod - state vector at a short timestep forwards
			datetime - datetime object

		Returns:
			A - Jacobian Matrix
		'''
		#Calculating right ascension and declination
		rai, deci = stateVectorToRaDec(Xi,datetime)
		raMod, decMod = stateVectorToRaDec(Xmod,datetime)

		#Changes in state and angles
		dRa = raMod - rai
		dDec = decMod - deci

		#Determine which state vector index is being used
		idx = np.nonzero(dX)[0][0]

		#Calculate Finite Differencing based on index
		if idx == 0:
			A = [[dRa/dX[idx],0.0,0.0,0.0,0.0,0.0],
				[dDec/dX[idx],0.0,0.0,0.0,0.0,0.0]]
		elif idx == 1:
			A = [[0.0,dRa/dX[idx],0.0,0.0,0.0,0.0],
				[0.0,dDec/dX[idx],0.0,0.0,0.0,0.0]]
		elif idx == 2:
			A = [[0.0,0.0,dRa/dX[idx],0.0,0.0,0.0],
				[0.0,0.0,dDec/dX[idx],0.0,0.0,0.0]]
		elif idx == 3:
			A = [[0.0,0.0,0.0,dRa/dX[idx],0.0,0.0],
				[0.0,0.0,0.0,dDec/dX[idx],0.0,0.0]]
		elif idx == 4:
			A = [[0.0,0.0,0.0,0.0,dRa/dX[idx],0.0],
				[0.0,0.0,0.0,0.0,dDec/dX[idx],0.0]]
		elif idx == 5:
			A = [[0.0,0.0,0.0,0.0,0.0,dRa/dX[idx]],
				[0.0,0.0,0.0,0.0,0.0,dDec/dX[idx]]]
		else:
			A = [[0.0,0.0,0.0,0.0,0.0,0.0],
				[0.0,0.0,0.0,0.0,0.0,0.0]]

		return A

	def calculateResiduals(self,Xi,raMeasure,decMeasure,datetime):
		'''Calculates the residuals between the current state vector
		and the observed values

		Attributes:
			Xi - state vector at current timestep
			latitude - latitude of ground station in degrees
			longitude - longitude of ground station in degrees
			altitude - altitude above sea level of ground station in meters
			datetime - datetime object

		Returns:
			b - measurement residuals
		'''
		#Calculating right ascension and declination
		rai, deci = stateVectorToRaDec(Xi,datetime)

		#Residual Matrix
		b = [raMeasure-rai, decMeasure - deci]

		return b

	def calculateCovarianceMatricies(self,A,b):
		'''Calculates the Covariance Matrix and Residual Matrix
		for the current timestep

		Attributes:
			a - jacobian maitrix
			b - measurement residuals
			W - weighting matrix

		Returns:
			AtWA - covariance matrix
			AtWb - residual matrix
			btWb - root mean square residuals
		'''	
		#Calculate Covariance Matrix
		AtWA = np.matmul(np.matmul(np.transpose(A),self.measurementWeighting),A)

		#Calculate Residual Matrix
		AtWb = np.matmul(np.matmul(np.transpose(A),self.measurementWeighting),b)

		#Calculate root mean square of the residuals
		btWb = np.matmul(np.matmul(np.transpose(b),self.measurementWeighting),b)

		return AtWA, AtWb, btWb



###########################################################################################################################

def statisticalOrbitDetermination(noradID,detectionRequestID,initialOrbit,numberOfPasses,measurementSpacing):

    #Set up Propagator
    initialEstimate = createPropagator(initialOrbit,mass,area,cd,cr)
    currentEstimate = createPropagator(initialOrbit,mass,area,cd,cr)

    #Initialize Tracking Variables
    trackRMS = []
    trackIter = []
    iteration = 0
    measurement = 0

    #Initialize  RMS values to enter loop
    RMSold = 0
    RMSnew = 1
    dRMS = RMSnew - RMSold

    #Number of Iterations of Batch Least Squares
    while np.linalg.norm(dRMS) >= TOLERANCE and iteration < MAX_ITERATIONS:
    #for k in range(0,1):
        #Initialize Tracking Variables
        raGuess = []
        decGuess = []
        raMeasure = []
        decMeasure = []
        measurementNumber = []
        sumAtWA = []
        sumAtWb = []
        sumb = []
        time = []

        #For each Orbital Pass being processed
        for j in range(0,numberOfPasses):
            #Process Next Orbital Pass
            orbitalPass = Pass(noradID, detectionRequestID[j])
            az, el, ra, dec, datetime, slantRange, rSite = orbitalPass.getConstantSpacing(measurementSpacing)

            #For each measurement in th Orbital Pass
            for i in range(0,len(datetime)):

                #For each dimesion in the State Fector
                for z in range(0,6):
                    #Modify the State for current Dimension
                    currentEstimate, __ = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)
                    modifiedEstimate, dX = modifyPropagator(z,initialEstimate,mass,area,cd,cr)

                    #Propagating State to measurement
                    pvi = currentEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
                    Xi = PVCoordinatesToList(pvi)

                    #Modified State
                    pvMod = modifiedEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
                    Xmod = PVCoordinatesToList(pvMod)

                    #Calculate Matricies
                    A = calculateFiniteDifference(Xi,Xmod,dX,datetime[i])
                    b = calculateResiduals(Xi,ra[i],dec[i],datetime[i]) 
                    AtWA, AtWb, btWb = calculateCovarianceMatricies(A,b)

                    #Sum Matricies for each measurement
                    if time == []:
                        sumAtWA = AtWA
                        sumAtWb = AtWb
                        sumbtWb = btWb
                    else:
                        sumAtWA = np.add(sumAtWA,AtWA)
                        sumAtWb = np.add(sumAtWb,AtWb)
                        sumbtWb = np.add(sumbtWb,btWb)

                    #Reset Propagators to initial estimate
                    currentEstimate = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)
                    modifiedEstimate = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)

                #Right Ascension and Declination of current state
                rai, deci = stateVectorToRaDec(Xi,datetime[i])

                #Tracking Variables
                raMeasure.append(ra[i])
                decMeasure.append(dec[i])
                raGuess.append(rai)
                decGuess.append(deci)
                time.append(datetime[i])

        #Average the Root Mean Square of residuals
        btWbAvg = np.divide(sumbtWb,len(time))

        #Inverse of AtWA matrix
        u, s, v = np.linalg.svd(sumAtWA,full_matrices=True)

        #Calculate Covariance Matrix
        P = np.matmul(np.matmul(np.transpose(v),np.linalg.inv(np.diag(s))),np.transpose(u))
        eigval, eigvect = np.linalg.eig(P[0:3,0:3])

        #Calculate Root Mean Square for iteration
        RMS = sqrt(btWbAvg/(2*len(time)))
        
        #Change in Initial State
        dX = np.matmul(P,sumAtWb)
        initialEstimate = updatePropagator(dX,initialEstimate,mass,area,cd,cr)

        #Compare the RMS with previous iteration for exit condition
        RMSold = RMSnew
        RMSnew = RMS
        dRMS = RMSnew - RMSold

        #Counter and Tracking Variables
        iteration += 1
        trackRMS.append(RMS)
        trackIter.append(iteration)

        #Status Update
        if iteration == 1 or iteration == 5 or iteration == 10:
            yn = userInput('Change in RMS is currently '+format(absolute(dRMS), "5.1e")+' at '+str(iteration)+' iterations. Continue? (y/n): ')
            if yn == 'n':
                break
            elif yn == 'y':
                continue 
            else:
                error('invalid option')
        else:
            statusUpdate(iteration)

    measurementNumber = []
    for i in range(0,len(raGuess)):
        measurementNumber.append(i)

    finalOrbit = initialEstimate.getInitialState().getOrbit()

    return raMeasure, raGuess, decMeasure, decGuess, time, trackRMS, trackIter, measurementNumber, initialEstimate, finalOrbit

def calculateFiniteDifference(Xi,Xmod,dX,datetime):
	'''Calculates the Jacobian Matrix of the observations
	with the current state vector

	Attributes:
		Xi - state vector at current timestep
		Xmod - state vector at a short timestep forwards
        datetime - datetime object

	Returns:
		A - Jacobian Matrix
	'''
	#Calculating right ascension and declination
	rai, deci = stateVectorToRaDec(Xi,datetime)
	raMod, decMod = stateVectorToRaDec(Xmod,datetime)

	#Changes in state and angles
	dRa = raMod - rai
	dDec = decMod - deci

	#Determine which state vector index is being used
	idx = np.nonzero(dX)[0][0]

	#Calculate Finite Differencing based on index
	if idx == 0:
		A = [[dRa/dX[idx],0.0,0.0,0.0,0.0,0.0],
			[dDec/dX[idx],0.0,0.0,0.0,0.0,0.0]]
	elif idx == 1:
		A = [[0.0,dRa/dX[idx],0.0,0.0,0.0,0.0],
			[0.0,dDec/dX[idx],0.0,0.0,0.0,0.0]]
	elif idx == 2:
		A = [[0.0,0.0,dRa/dX[idx],0.0,0.0,0.0],
			[0.0,0.0,dDec/dX[idx],0.0,0.0,0.0]]
	elif idx == 3:
		A = [[0.0,0.0,0.0,dRa/dX[idx],0.0,0.0],
			[0.0,0.0,0.0,dDec/dX[idx],0.0,0.0]]
	elif idx == 4:
		A = [[0.0,0.0,0.0,0.0,dRa/dX[idx],0.0],
			[0.0,0.0,0.0,0.0,dDec/dX[idx],0.0]]
	elif idx == 5:
		A = [[0.0,0.0,0.0,0.0,0.0,dRa/dX[idx]],
			[0.0,0.0,0.0,0.0,0.0,dDec/dX[idx]]]
	else:
		A = [[0.0,0.0,0.0,0.0,0.0,0.0],
			[0.0,0.0,0.0,0.0,0.0,0.0]]

	return A

def calculateResiduals(Xi,raMeasure,decMeasure,datetime):
	'''Calculates the residuals between the current state vector
	and the observed values

	Attributes:
		Xi - state vector at current timestep
        latitude - latitude of ground station in degrees
        longitude - longitude of ground station in degrees
        altitude - altitude above sea level of ground station in meters
        datetime - datetime object

	Returns:
		b - measurement residuals
	'''
	#Calculating right ascension and declination
	rai, deci = stateVectorToRaDec(Xi,datetime)

	#Residual Matrix
	b = [raMeasure-rai, decMeasure - deci]

	return b

def calculateCovarianceMatricies(A,b,W = [[1,0],[0,1]]):
	'''Calculates the Covariance Matrix and Residual Matrix
	for the current timestep

	Attributes:
		a - jacobian maitrix
		b - measurement residuals
		W - weighting matrix

	Returns:
		AtWA - covariance matrix
		AtWb - residual matrix
		btWb - root mean square residuals
	'''	
	#Calculate Covariance Matrix
	AtWA = np.matmul(np.matmul(np.transpose(A),W),A)

	#Calculate Residual Matrix
	AtWb = np.matmul(np.matmul(np.transpose(A),W),b)

	#Calculate root mean square of the residuals
	btWb = np.matmul(np.matmul(np.transpose(b),W),b)

	return AtWA, AtWb, btWb










