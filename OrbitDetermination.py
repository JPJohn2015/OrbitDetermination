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
from Variables import *

def createPropagator(orbit,mass,area,cd,cr):
	'''Creates Orekit Propagator for the specified orbit

	Attributes:
		orbit - orbit object 
		mass - estimated mass of satellite
		area - estimated cross sectional area of satellite
		cd - estimated coefficient of drag of satellite
		cr - estimated coefficient of reflectivity of satellite

	Returns:
		currentEstimate - object propagator
	'''
	#Defining Tolerances for propagator
	orbitType = OrbitType.CARTESIAN
	tol = NumericalPropagator.tolerances(POSITION_TOLERANCE, orbit, orbitType)

	#Defining Integrator 
	integrator = DormandPrince853Integrator(MIN_STEP, MAX_STEP, 
	    JArray_double.cast_(tol[0]),
	    JArray_double.cast_(tol[1]))
	integrator.setInitialStepSize(INIT_STEP)

	#Defining initial Spacecraft State and Orbit
	initialState = SpacecraftState(orbit, mass)
	propagator = NumericalPropagator(integrator)
	propagator.setOrbitType(orbitType)
	propagator.setInitialState(initialState)

	#Adding Perturbations: Earth Gravity Field
	gravityProvider = GravityFieldFactory.getNormalizedProvider(16, 16)
	gravityAttractionModel = HolmesFeatherstoneAttractionModel(EARTH.getBodyFrame(), gravityProvider)
	propagator.addForceModel(gravityAttractionModel)

	#Adding Perturbations: 3rd Body
	moon_3dbodyattraction = ThirdBodyAttraction(MOON)
	propagator.addForceModel(moon_3dbodyattraction)
	sun_3dbodyattraction = ThirdBodyAttraction(SUN)
	propagator.addForceModel(sun_3dbodyattraction)

	'''
	#Adding Perturbations: Atmospheric Drag
	atmosphere = SimpleExponentialAtmosphere(WGS84ELLIPSOID, DENSITY, HEIGHT_SEA, SCALE_HEIGHT)
	isotropicDrag = IsotropicDrag(area, cd)
	dragForce = DragForce(atmosphere, isotropicDrag)
	propagator.addForceModel(dragForce)

	#Adding Perturbations: Solar radiation pressure
	isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(area, cr);
	solarRadiationPressure = SolarRadiationPressure(SUN, WGS84ELLIPSOID.getEquatorialRadius(),
	                                                isotropicRadiationSingleCoeff)
	propagator.addForceModel(solarRadiationPressure)
	'''

	return propagator

def createSGP4Propagator(tle,mass,area,cd,cr,initialDate):
	'''Creates Orekit SGP4 Propagator for the specified TLE

	Attributes:
		tle - tle object
		mass - estimated mass of satellite
		area - estimated cross sectional area of satellite
		cd - estimated coefficient of drag of satellite
		cr - estimated coefficient of reflectivity of satellite
		initialDate - datetime to propagate tle

	Returns:
		currentEstimate - object propagator
	'''
	#Satellite Orientation
	nadirPointing = NadirPointing(ECI, WGS84ELLIPSOID)
	
	#Defining initial Spacecraft State and Orbit
	sgp4Propagator = SGP4(tle, nadirPointing, float(mass))
	tleInitialState = sgp4Propagator.getInitialState()
	tleEpoch = tleInitialState.getDate()
	tleOrbit_TEME = tleInitialState.getOrbit()
	tlePV_ECI = tleOrbit_TEME.getPVCoordinates(ECI)
	tleOrbit_ECI = KeplerianOrbit(tlePV_ECI, ECI, WGS84ELLIPSOID.getGM())

	#Defining Tolerances for propagator
	orbitType = OrbitType.CARTESIAN
	tol = NumericalPropagator.tolerances(POSITION_TOLERANCE, tleOrbit_ECI, orbitType)

	#Defining Integrator 
	integrator = DormandPrince853Integrator(MIN_STEP, MAX_STEP, 
	    JArray_double.cast_(tol[0]),
	    JArray_double.cast_(tol[1]))
	integrator.setInitialStepSize(INIT_STEP)

	#Defining initial Spacecraft State and Orbit
	initialState = SpacecraftState(tleOrbit_ECI, mass)
	propagator = NumericalPropagator(integrator)
	propagator.setOrbitType(orbitType)
	propagator.setInitialState(initialState)

	#Adding Perturbations: Earth Gravity Field
	gravityProvider = GravityFieldFactory.getNormalizedProvider(16, 16)
	gravityAttractionModel = HolmesFeatherstoneAttractionModel(EARTH.getBodyFrame(), gravityProvider)
	propagator.addForceModel(gravityAttractionModel)

	#Adding Perturbations: 3rd Body
	moon_3dbodyattraction = ThirdBodyAttraction(MOON)
	propagator.addForceModel(moon_3dbodyattraction)
	sun_3dbodyattraction = ThirdBodyAttraction(SUN)
	propagator.addForceModel(sun_3dbodyattraction)

	'''
	#Adding Perturbations: Atmospheric Drag
	atmosphere = SimpleExponentialAtmosphere(WGS84ELLIPSOID, DENSITY, HEIGHT_SEA, SCALE_HEIGHT)
	isotropicDrag = IsotropicDrag(area, cd)
	dragForce = DragForce(atmosphere, isotropicDrag)
	propagator.addForceModel(dragForce)

	#Adding Perturbations: Solar radiation pressure
	isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(area, cr);
	solarRadiationPressure = SolarRadiationPressure(SUN, WGS84ELLIPSOID.getEquatorialRadius(),
	                                                isotropicRadiationSingleCoeff)
	propagator.addForceModel(solarRadiationPressure)
	'''
	
	#Propagate to IOD Epoch
	propagator.propagate(initialDate).getPVCoordinates()

	return propagator

def updatePropagator(dX,initialEstimate,mass,area,cd,cr):
	'''Creates a new propagator object with the updated state

	Attributes:
		dX - recommended change in state
		currentEstimate - current propagator
        latitude - latitude of ground station in degrees
        longitude - longitude of ground station in degrees
        altitude - altitude above sea level of ground station in meters
        datetime - datetime object

	Returns:
		newEstimate - new propagator
	'''	
	#Get initial State to update
	currentOrbit = initialEstimate.getInitialState().getOrbit()
	X = PVCoordinatesToList(currentOrbit.getPVCoordinates())

	#Get State Epoch
	date = currentOrbit.getPVCoordinates().getDate()

	#Update State
	Xnew = np.add(X,dX)

	#Create new PVCoordinates for new State
	position = Vector3D(float(Xnew[0]),float(Xnew[1]),float(Xnew[2]))
	velocity = Vector3D(float(Xnew[3]),float(Xnew[4]),float(Xnew[5]))
	pvNew = TimeStampedPVCoordinates(date,position,velocity)

	#Create Orbit and Propagator
	keplerianOrbit = KeplerianOrbit(pvNew,ECI,EARTH_MU)
	currentEstimate = createPropagator(keplerianOrbit,mass,area,cd,cr)

	return currentEstimate

def modifyPropagator(i,initialEstimate,mass,area,cd,cr):
	'''Creates a new propagator object with the updated state

	Attributes:
		i - integer that determines which dimension is being modified
		initialEstimate - current propagator
		latitude - latitude of ground station in degrees
        longitude - longitude of ground station in degrees
        altitude - altitude above sea level of ground station in meters
        datetime - datetime object

	Returns:
		newEstimate - new propagator
		dX - change in modified state
	'''	
	#Get Initial State to modify
	currentOrbit = initialEstimate.getInitialState().getOrbit()
	X = PVCoordinatesToList(currentOrbit.getPVCoordinates())

	#Get State Epoch
	date = currentOrbit.getPVCoordinates().getDate()

	#Modify the required dimesion of the State
	if i == 0:
		dX = [X[0]*0.001,0.0,0.0,0.0,0.0,0.0]
	elif i == 1:
		dX = [0.0,X[1]*0.001,0.0,0.0,0.0,0.0]
	elif i == 2:
		dX = [0.0,0.0,X[2]*0.001,0.0,0.0,0.0]
	elif i == 3:
		dX = [0.0,0.0,0.0,X[3]*0.001,0.0,0.0]
	elif i ==4:
		dX = [0.0,0.0,0.0,0.0,X[4]*0.001,0.0]
	elif i == 5:
		dX = [0.0,0.0,0.0,0.0,0.0,X[4]*0.001]
	else:
		dX = [0.0,0.0,0.0,0.0,0.0,0.0]

	#Modify State
	Xnew = np.add(X,dX)

	#Create new PVCoordinates for new State
	position = Vector3D(float(Xnew[0]),float(Xnew[1]),float(Xnew[2]))
	velocity = Vector3D(float(Xnew[3]),float(Xnew[4]),float(Xnew[5]))
	pvNew = TimeStampedPVCoordinates(date,position,velocity)

	#Create Orbit and Propagator
	keplerianOrbit = KeplerianOrbit(pvNew,ECI,EARTH_MU)
	currentEstimate = createPropagator(keplerianOrbit,mass,area,cd,cr)

	return currentEstimate, dX

def circularEstimate(noradID,detectionRequestID):

	#Generate Pass Objects
	pass1 = Pass(noradID,detectionRequestID[0])
	pass2 = Pass(noradID,detectionRequestID[1])

	#Calculate Time separation between passes
	deltaTime1 = (pass2.initialEpoch - pass1.initialEpoch).total_seconds()
	deltaTime2 = (pass2.finalEpoch - pass1.finalEpoch).total_seconds()
	deltaTime3 = (pass2.middleEpoch - pass1.middleEpoch).total_seconds()
	dtAvg = (deltaTime1 + deltaTime2 + deltaTime3)/3

	#Load Data from First Pass
	az, el, ra, dec, datetime, slantRange, rSite = pass1.getThreePointCentered(pass1.getPassLength()/2)

	#Determine Maximum number of Revolutions possible
	minAltitude = 100
	minPeriod = 2*pi*sqrt(((minAltitude + EARTH_RADIUS_KM)**3)/EARTH_MU_KM)
	maxRevs = ceil(dtAvg/minPeriod)

	#Initialize Tracking Variables
	orbitList = []
	errorList = []
	smaList = []

	#Check orbit for each possible revolution number
	for n in range(1,maxRevs):
		#Determine Integer Period 
		period = dtAvg/n

		#Calculate Semi-Major Axis and Altitude
		semiMajorAxis = (EARTH_MU_KM*(period/(2*pi))**2)**(1/3)
		altitude = semiMajorAxis - EARTH_RADIUS_KM

		#Guess the slant Range Magnitude based on altitude
		Re = EARTH_RADIUS_KM
		rho = []
		for i in range(0,len(el)):
			#Range equation for circular orbits
			d = Re*(sqrt(((altitude+Re)/Re)**2 - cos(radians(el[i]))**2) - sin(radians(el[i])))
			rho.append(d)

		#Gooding Initial Orbit Determination for better estimate
		iod = IodGooding(ECI, EARTH_MU)
		orbit = iod.estimate(rSite[0],rSite[1],rSite[2],
	            slantRange[0],datetime_to_absolutedate(datetime[0]),
	            slantRange[1],datetime_to_absolutedate(datetime[1]),
	            slantRange[2],datetime_to_absolutedate(datetime[2]),
				float(rho[0]*1000),float(rho[2]*1000))

		#Error between circular semi-major axis and IOD semi-major axis
		error = np.absolute(semiMajorAxis - orbit.getA()/1000)

		#Append Tracking Variables
		smaList.append(semiMajorAxis)
		orbitList.append(orbit)
		errorList.append(error)

	#Index with minimum semi-major axis error
	idx = errorList.index(min(errorList))
	referenceOrbit = orbitList[idx]
	initialDate = referenceOrbit.getDate()

	#Initial Orbit Determination Guess with defined eccentricity
	initialOrbit = KeplerianOrbit(smaList[idx]*1000, 0.01, 
				   referenceOrbit.getI(), 
				   referenceOrbit.getPerigeeArgument(), 
				   referenceOrbit.getRightAscensionOfAscendingNode(), 
				   referenceOrbit.getTrueAnomaly(), 
				   PositionAngle.TRUE, ECI, 
				   initialDate, 
				   EARTH_MU)

	return initialOrbit

def initialOrbitDetermination(noradID,detectionRequestID,method):
	'''Initial Orbit Determination from angle's only data

	Attributes:
		noradID - NORAD ID of the satellite
		passNumber - number of the pass being tested

	Returns:
		orbit - orbit object of initial orbit etermination

	'''
	#NORAD ID and Detection Request ID
	initialPass = Pass(noradID, detectionRequestID[0])

	#Process Dataframe into lists for initial orbit determination
	az, el, ra, dec, time, slantRange, rSite = initialPass.getThreePointCentered(0.98*initialPass.getPassLength()/2)

	if method == 'doubleR':      
		#Double-R Method for preliminary guess for range values
		_, rho1, rho2, rho3, _, _ = doubleRMethod(rSite[0],rSite[1],rSite[2],
					slantRange[0],time[0],
					slantRange[1],time[1],
					slantRange[2],time[2])

		#Gooding Initial Orbit Determination for better estimate
		iod = IodGooding(ECI, EARTH_MU)
		initialOrbit = iod.estimate(rSite[0],rSite[1],rSite[2],
	            slantRange[0],datetime_to_absolutedate(time[0]),
	            slantRange[1],datetime_to_absolutedate(time[1]),
	            slantRange[2],datetime_to_absolutedate(time[2]),
				float(rho1*KM2M),float(rho3*KM2M))

	elif method == 'gauss':
		#Double-R Method for preliminary guess for range values
		_, rho1, rho2, rho3, _, _ = gaussMethod(rSite[0],rSite[1],rSite[2],
					slantRange[0],time[0],
					slantRange[1],time[1],
					slantRange[2],time[2])

		#Gooding Initial Orbit Determination for better estimate
		iod = IodGooding(ECI, EARTH_MU)
		initialOrbit = iod.estimate(rSite[0],rSite[1],rSite[2],
	            slantRange[0],datetime_to_absolutedate(time[0]),
	            slantRange[1],datetime_to_absolutedate(time[1]),
	            slantRange[2],datetime_to_absolutedate(time[2]),
				float(rho1*KM2M),float(rho3*KM2M))

	elif method == 'circular':
		#Circular Orbit Assumption
		initialOrbit = circularEstimate(noradID,detectionRequestID)
	else:
		error('Invalid IOD Method')

	return initialOrbit

def doubleRMethod(rSite1,rSite2,rSite3,slantRange1,datetime1,slantRange2,datetime2,slantRange3,datetime3):
	'''Calculates a position and velocity vector in earth
	centered inertial for the middle observation.

    Attributes:
        M1 - first chronological Measurement object
        M2 - second chronological Measurement object
        M3 - third chronological Measurement object

    Returns:
    	r2vec - position vector for second measurement
    	v2vec - velocity vector for second measurement
    '''
    #Format Slant Range Vectors
	rho1 = [slantRange1.getX(),slantRange1.getY(),slantRange1.getZ()]
	rho2 = [slantRange2.getX(),slantRange2.getY(),slantRange2.getZ()]
	rho3 = [slantRange3.getX(),slantRange3.getY(),slantRange3.getZ()]

	#Format Site Vectors
	R1 = np.multiply([rSite1.getX(),rSite1.getY(),rSite1.getZ()],0.001)
	R2 = np.multiply([rSite2.getX(),rSite2.getY(),rSite2.getZ()],0.001)
	R3 = np.multiply([rSite3.getX(),rSite3.getY(),rSite3.getZ()],0.001)

	#Format Time Separation
	tau1 = (datetime1 - datetime2).total_seconds()
	tau3 = (datetime3 - datetime2).total_seconds()

    #Initial guess for iteration
	r1 = 2*EARTH_RADIUS_KM
	r2 = 2.01*EARTH_RADIUS_KM
	c1 = np.dot(np.multiply(2,rho1),R1)
	c2 = np.dot(np.multiply(2,rho2),R2)

    #Iteration
	error = 1
	counter = 0
	while error > 0.001:
		counter += 1
		F1,F2,f,g,r3vec,r2vec,q1,q2,q3 = dr_function(c1,c2,r1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)
		dr1 = np.multiply(0.005,r1)
		dr2 = np.multiply(0.005,r2)
		F1r1dr1,F2r1dr1,_,_,_,_,_,_,_ = dr_function(c1,c2,r1+dr1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)
		F1r2dr2,F2r2dr2,_,_,_,_,_,_,_ = dr_function(c1,c2,r1,r2+dr2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)

		dF1dr1 = (F1r1dr1 - F1)/dr1
		dF2dr1 = (F2r1dr1 - F2)/dr1
		dF1dr2 = (F1r2dr2 - F1)/dr2
		dF2dr2 = (F2r2dr2 - F2)/dr2
		
		delta = dF1dr1*dF2dr2 - dF2dr1*dF1dr2
		delta1 = dF2dr2*F1 - dF1dr2*F2
		delta2 = dF1dr1*F2 - dF2dr1*F1
		dr1 = -delta1/delta
		dr2 = -delta2/delta
    
    	#Error calculation
		error = (np.absolute(dr1) + np.absolute(dr2))/2;
		r1 = r1 + dr1
		r2 = r2 + dr2

	#Velocity vector
	v2vec, v3vec = lamberts_problem(r2vec,r3vec,tau3,-1)

	pv = TimeStampedPVCoordinates(datetime_to_absolutedate(datetime2),
			Vector3D(float(r2vec[0]*KM2M),float(r2vec[1]*KM2M),float(r2vec[2]*KM2M)),
			Vector3D(float(v2vec[0]*KM2M),float(v2vec[1]*KM2M),float(v2vec[2]*KM2M)))
	initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

	return initialOrbit, q1, q2, q3, r2vec, v2vec

def dr_function(c1,c2,r1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3):
	'''Sub-function that calculates lagrangian multipliers 
	as well as other function values for the main iteration.

	Attributes:
        c - angle constant
        r - position vector magnitude
        tau - time difference
        rho - slant range vector
        R - site vector

    Returns:
    	F - time constant
    	f,g - lagrange multipliers
    	rvec - position vector
	'''
	#Slant Range 1 and 2 magnitude
	q1 = (-c1 + np.sqrt((c1**2) - 4*(np.dot(R1,R1) - (r1**2))))/2
	q2 = (-c2 + np.sqrt((c2**2) - 4*(np.dot(R2,R2) - (r1**2))))/2

	#Measurement 1 and 2 position vectors
	r1vec = R1 + np.multiply(q1,rho1)
	r1 = norm(r1vec)
	r2vec = R2 + np.multiply(q2,rho2)
	r1 = norm(r2vec)

	#Measurement 3 slant range and position vector
	wvec = np.cross(r1vec,r2vec)/(norm(r1vec)*norm(r2vec))
	q3 = np.dot(-R3,wvec)/np.dot(rho3,wvec)
	r3vec = R3 + np.multiply(q3,rho3)
	r3 = norm(r3vec)
       
    #Cosine angles between position vectors 
	cos21 = np.dot(r2vec,r1vec)/(norm(r2vec)*norm(r1vec))
	t21 = np.arccos(cos21)
	cos31 = np.dot(r3vec,r1vec)/(norm(r3vec)*norm(r1vec))
	t31 = np.arccos(cos31);
	cos32 = np.dot(r3vec,r2vec)/(norm(r3vec)*norm(r2vec));
	t32 = np.arccos(cos32);

	#Sine angles between position vectors
	sin21 = np.sqrt(1-(cos21**2))
	sin31 = np.sqrt(1-(cos31**2))
	sin32 = np.sqrt(1-(cos31**2))
    
    #Calculate other constants   
	theta31 = np.arccos(cos31)
	if theta31 > pi/2:
		c1 = (r2*sin32)/(r1*sin31)
		c3 = (r2*sin21)/(norm(r3vec)*sin31)
		p = (c1*r1 + c3*r3 - r2)/(c1 + c3 - 1)
	else:
		c1 = (r1*sin31)/(r2*sin32)
		c3 = (r1*sin21)/(r3*sin32)
		p = (c3*r3 - c1*r2 + r1)/(-c1 + c3 + 1)
    
    #More angles and parameters  
	ecos1 = (p/r1) - 1
	ecos2 = (p/r2) - 1
	ecos3 = (p/r3) - 1
	theta21 = np.arccos(cos21)
	if theta21 != pi/2:
		esin2 = (-cos21*ecos2 + ecos1)/sin21;
	else:
		esin2 = (cos32*ecos2 - ecos3)/sin31;
	e2 = (ecos2**2) + (esin2**2)
	a = p/(1-e2)

	#Final lagrange multipliers and constants based on eccentricity
	if np.sqrt(e2) < 1:
		n = np.sqrt(EARTH_MU_KM/(a**3))
		S = r2/p*np.sqrt(1-e2)*esin2
		C = r2/p*(e2 + ecos2)
		sinE32 = r3/np.sqrt(a*p)*sin32 - r3/(p*(1 - cos32)*S)
		cosE32 = 1 - (r2*r3)/(a*p)*(1-cos32)
		sinE21 = r1/np.sqrt(a*p)*sin21 + r1/(p*(1 - cos21)*S)
		cosE21 = 1 - (r2*r1)/(a*p)*(1-cos21)
		M32 = np.arccos(cosE32) + 2*S*(np.sin(np.arccos(cosE32)/2)**2) - C*sinE32
		M12 = -np.arccos(cosE21) + 2*S*(np.sin(np.arccos(cosE21)/2)**2) + C*sinE21
		F1 = tau1 - M12/n
		F2 = tau3 - M32/n
		f = 1 - (a/r2)*(1 - cosE32)
		g = tau3 - np.sqrt((a**3)/EARTH_MU_KM)*(np.arccos(cosE32) - sinE32)      
	else:
		n = np.sqrt(EARTH_MU_KM/(-a)**3)
		Sh = r2/p*np.sqrt(e2-1)*esin2
		Ch = r2/p*(e2 + ecos2)
		sinhF32 = r3/np.sqrt(-a*p)*sin32 - r3/(p*(1 - cos32)*Sh)
		F32 = np.log(sinhF32 + np.sqrt(sinhF32 + 1))
		sinhF21 = r1/np.sqrt(-a*p)*sin21 + r1/(p*(1 - cos32)*Sh)
		F21 = np.log(sinhF21 + np.sqrt(sinhF21**2 + 1))
		M32 = -F32 + 2*Sh*np.sinh(F32/2)**2 + Ch*sinhF32
		M12 = F21 + 2*Sh*np.sinh(F21/2)**2 + Ch*sinhF21
		F1 = tau1 - M12/n
		F2 = tau3 - M32/n
		f = 1 - (-a/r2)*(1 - np.cosh(F32))
		g = tau3 - np.sqrt((-a)**3/EARTH_MU_KM)*(F32 - sinhF32)

	return F1, F2, f, g, r3vec, r2vec, q1, q2, q3

def gaussMethod(rSite1,rSite2,rSite3,slantRange1,datetime1,slantRange2,datetime2,slantRange3,datetime3):
	'''Calculates a position and velocity vector in earth
	centered inertial for the middle observation.

    Attributes:
        M1 - first chronological Measurement object
        M2 - second chronological Measurement object
        M3 - third chronological Measurement object

    Returns:
    	r2vec - position vector for second measurement
    	v2vec - velocity vector for second measurement
    '''
    #Format Slant Range Vectors
	rho1 = np.array([slantRange1.getX(),slantRange1.getY(),slantRange1.getZ()])
	rho2 = np.array([slantRange2.getX(),slantRange2.getY(),slantRange2.getZ()])
	rho3 = np.array([slantRange3.getX(),slantRange3.getY(),slantRange3.getZ()])

	#Format Site Vectors
	R1 = np.multiply([rSite1.getX(),rSite1.getY(),rSite1.getZ()],M2KM)
	R2 = np.multiply([rSite2.getX(),rSite2.getY(),rSite2.getZ()],M2KM)
	R3 = np.multiply([rSite3.getX(),rSite3.getY(),rSite3.getZ()],M2KM)

	#Format Time Separation
	tau1 = (datetime1 - datetime2).total_seconds()
	tau3 = (datetime3 - datetime2).total_seconds()
	
	#Time constant
	tau = tau3 - tau1

	#Vector cross products
	q1vec = np.cross(rho2,rho3)
	q2vec = np.cross(rho1,rho3)
	q3vec = np.cross(rho1,rho2)
	Do = np.dot(rho1,q1vec)

	#Scalar matrix
	D = [[np.dot(R1,q1vec),np.dot(R1,q2vec),np.dot(R1,q3vec)],
		[np.dot(R2,q1vec),np.dot(R2,q2vec),np.dot(R2,q3vec)],
		[np.dot(R3,q1vec),np.dot(R3,q2vec),np.dot(R3,q3vec)]]

	#Slant range at position 2
	A = (1/Do)*(-D[0][1]*(tau3/tau) + D[1][1] + D[2][1]*(tau1/tau))
	B = (1/(6*Do))*(D[0][1]*(tau3**2 - tau**2)*(tau3/tau) + D[2][1]*(tau**2 - tau1**2)*(tau1/tau))
	E = np.dot(R2,rho2)
	a = -((A**2) + (2*A*E) + np.dot(R2,R2))
	b = -2*EARTH_MU_KM*B*(A + E)
	c = -(EARTH_MU_KM**2)*(B**2)

	#Solving roots of characteristic equation
	coeff = [1, 0, a, 0, 0, b, 0, 0, c]
	r = np.roots(coeff)
	posRealRoot = max(r[np.isreal(r)]).real
	xn = posRealRoot

	#Magnitude of slant ranges
	q1 = 1/Do*((6*(D[2][0]*(tau1/tau3) + D[1][0]*(tau/tau3))*(xn**3) + \
		EARTH_MU_KM*D[2][0]*(tau**2 - tau1**2)*(tau1/tau3))/(6*(xn**3) + EARTH_MU_KM*(tau**2 - tau3**2)) - D[0][0])
	q2 = A + EARTH_MU_KM*B/(xn**3)
	q3 = 1/Do*((6*(D[0][2]*(tau3/tau1) - D[1][2]*(tau/tau1))*(xn**3) + \
		EARTH_MU_KM*D[0][2]*(tau**2 - tau3**2)*(tau3/tau1))/(6*(xn**3) + EARTH_MU_KM*(tau**2 - tau1**2)) - D[2][2])

	#Position vectors
	r1vec = R1 + np.multiply(rho1,q1)
	r2vec = R2 + np.multiply(rho2,q2)
	r3vec = R3 + np.multiply(rho3,q3)

	#Lagrange Multipliers
	f1 = 1 - 0.5*EARTH_MU_KM*((tau1**2)/(norm(r2vec)**3))
	f3 = 1 - 0.5*EARTH_MU_KM*((tau3**2)/(norm(r2vec)**3))
	g1 = tau1 - (1/6)*EARTH_MU_KM*((tau1**3)/(norm(r2vec)**3))
	g3 = tau3 - (1/6)*EARTH_MU_KM*((tau3**3)/(norm(r2vec)**3))

	#Velocity vector at position 2
	v2vec = 1/(f1*g3 - f3*g1)*(-f3*r1vec + f1*r3vec)

	print(r2vec)
	print(r3vec)

	lv2vec, lv3vec = lamberts_problem(r2vec,r3vec,tau3,-1)

	#Save first iteration variables
	q1_old = q1
	q2_old = q2
	q3_old = q3

	#Iteration set up variables
	err = 1
	n = 0
	nmax = 1000
	tol = 1e-6

	#Iterative improvement of gauss
	while err > tol and (n < nmax):
	#while (diff1 > tol) and (diff2 > tol) and (diff3 > tol) and (n < nmax):
		n += 1

		#Compute universal kepler's equation parameters
		ro = norm(r2vec)
		vo = norm(v2vec)
		vro = np.dot(v2vec,r2vec)/ro
		a = (2/ro) - (vo**2)/EARTH_MU_KM

		#Kepler equations
		_, _, ff1, gg1, _, _, _ = universal_variables(r2vec,v2vec,tau1)
		_, _, ff3, gg3, _, _, _ = universal_variables(r2vec,v2vec,tau3)
		
		#Average lagrange multipliers
		f1 = (f1 + ff1)/2
		f3 = (f3 + ff3)/2
		g1 = (g1 + gg1)/2
		g3 = (g3 + gg3)/2
		c1 = g3/(f1*g3 - f3*g1)
		c3 = -g1/(f1*g3 - f3*g1)

		#New slant range magnitudes
		q1 = 1/Do*( -D[0][0] + 1/c1*D[1][0] - c3/c1*D[2][0])
		q2 = 1/Do*( -c1*D[0][1] + D[1][1] - c3*D[2][1])
		q3 = 1/Do*(-c1/c3*D[0][2] + 1/c3*D[1][2] - D[2][2])

		#New position vectors
		r1vec = R1 + np.multiply(q1,rho1)
		r2vec = R2 + np.multiply(q2,rho2)
		r3vec = R3 + np.multiply(q3,rho3)

		#New velocity vector
		v2 = (-f3*r1vec + f1*r3vec)/(f1*g3 - f3*g1)

		#Error
		err = np.sqrt((q1 - q1_old)**2 + (q2 - q2_old)**2 + (q3 - q3_old)**2)

		#Update slant ranges
		q1_old = q1
		q2_old = q2
		q3_old = q3

	v2vec, v3vec = lamberts_problem(r2vec,r3vec,tau3,-1)

	pv = TimeStampedPVCoordinates(datetime_to_absolutedate(datetime2),
			Vector3D(float(r2vec[0]*KM2M),float(r2vec[1]*KM2M),float(r2vec[2]*KM2M)),
			Vector3D(float(v2vec[0]*KM2M),float(v2vec[1]*KM2M),float(v2vec[2]*KM2M)))
	initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

	return initialOrbit, q1, q2, q3, r2vec, v2vec

def lamberts_problem(r1,r2,dt,path):

	#https://www.mathworks.com/matlabcentral/fileexchange/44789-solve-lambert-s-problem-in-two-body-dynamics

	r1norm = np.linalg.norm(r1)
	r2norm = np.linalg.norm(r2)
	mutt = np.sqrt(EARTH_MU_KM)*dt
	dnu = np.arccos(np.dot(r1, r2)/(r1norm*r2norm))
	A = np.sqrt(r1norm*r2norm*(1+np.cos(dnu)))
	if path == 1:
		A = -A
	'''
	z = 0
	for i in range(0,50):
		S = stumpS(z)
		C = stumpC(z)
		y = np.absolute(r1norm+r2norm - A*(1-z*S)/np.sqrt(C))
		x = np.sqrt(y/C)
		t = (x**3)*S + A*np.sqrt(y)
		if np.absolute(t - mutt) < 1e-6:
			f = 1-y/r1norm
			g = A*np.sqrt(y/EARTH_MU_KM)
			gd = 1 -y/r2norm
			v1 = (r2 - f*r1)/g
			v2 = (gd*r2 - r1)/g
		if np.absolute(z) > 1e-6:
			Cp = (1-z*S-2*C)/(2*z)
			Sp = (C-3*S)/(2*z)
			tp = (x**3)*(Sp-1.5*S*Cp/C)+0.125*A*(3*S*np.sqrt(y)/C+A/x)
		else:
			tp = (np.sqrt(2)/40)*(y**1.5)+0.125*A*(np.sqrt(y)+A*np.sqrt(0.5/y))
		z = z-(t-mutt)/tp
	'''

	z = 0
	err = 1
	while err > 1e-6:
	#for i in range(0,50):
		S = stumpS(z)
		C = stumpC(z)
		y = np.absolute(r1norm+r2norm - A*(1-z*S)/np.sqrt(C))
		x = np.sqrt(y/C)
		t = (x**3)*S + A*np.sqrt(y)
		err = np.absolute(t - mutt)

		if np.absolute(z) > 1e-6:
			Cp = (1-z*S-2*C)/(2*z)
			Sp = (C-3*S)/(2*z)
			tp = (x**3)*(Sp-1.5*S*Cp/C)+0.125*A*(3*S*np.sqrt(y)/C+A/x)
		else:
			tp = (np.sqrt(2)/40)*(y**1.5)+0.125*A*(np.sqrt(y)+A*np.sqrt(0.5/y))

		f = 1-y/r1norm
		g = A*np.sqrt(y/EARTH_MU_KM)
		gd = 1 -y/r2norm
		v1 = (r2 - f*r1)/g
		v2 = (gd*r2 - r1)/g
		z = z-(t-mutt)/tp

	return v1, v2

def universal_variables(r2vec,v2vec,dt):
	'''Solves the lambert problem to propagate a state
	vector by some time.

    Attributes:
        r2vec - position vector
        v2vec - velocity vector
        dt - propagation time

    Returns:
    	rvec - position vector after time dt
    	vvec - velocity vector after time dt
    	f,g,fdot,gdot - lagrange multipliers
    	X - universal variables
    '''
	#Define working parameters
	ro = norm(r2vec)
	vo = norm(v2vec)
	vro = np.dot(r2vec,v2vec)/ro
	a = (2/ro) - (vo**2)/398600
	
	#Initial universal variable
	X = np.sqrt(398600)*np.absolute(a)*dt

	#Set error tolerance
	tol = 1e-8
	nmax = 5000

	#Newton's method for iteration
	ratio = 1
	m = 0
	count = 1

	#Iteration
	while (np.absolute(ratio) > tol) and (m < nmax):
		m += 1
		z = a*(X**2)
		funcX = ((ro*vro)/np.sqrt(398600))*(X**2)*stumpC(z) + (1-a*ro)*(X**3)*stumpS(z) + ro*X-np.sqrt(398600)*dt
		funcXdot = ((ro*vro)/np.sqrt(398600))*X*(1-a*(X**2)*stumpS(z)) + (1-a*ro)*(X**2)*stumpC(z)+ro
		ratio = funcX/funcXdot
		X = X - ratio
		count += 1
	z = a*(X**2)

	#Get new lagrange multipliers
	f = 1 - ((X**2)/ro)*stumpC(z)
	g = dt - (1/np.sqrt(398600))*(X**3)*stumpS(z)
	rvec = f*r2vec + g*v2vec
	r = norm(rvec)
	fdot = (np.sqrt(398600)/(r*ro))*(a*(X**3)*stumpS(z) - X)
	gdot = 1 - ((X**2)/r)*stumpC(z)
	vvec = fdot*r2vec + gdot*v2vec

	return rvec, vvec, f, g, fdot, gdot, X

def stumpS(z):
	'''Taylor series expansion of S Stumpff Function.

    Attributes:
        z - modified universal variable

    Returns:
    	S - stumpff function value
    '''
    #Taylor series expansion
	S = 0
	for k in range(1,10):
		S = S + (((-1)**(k-1))*(z**(k-1)))/np.math.factorial(2*(k-1) + 3)

	return S

def stumpC(z):
	'''Taylor series expansion of C Stumpff Function.

    Attributes:
        z - modified universal variable

    Returns:
    	C - stumpff function value
    '''
    #Taylor series expansion
	C = 0
	for k in range(1,10):
		C = C + (((-1)**(k-1))*(z**(k-1)))/np.math.factorial(2*(k-1) + 2)

	return C

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






