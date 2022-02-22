'''
Initial Orbit Determination Library has functions for performing
Initial Orbit Determination.

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

class InitialOrbitDetermination:

	def __init__(self, noradID, detectionRequestID, method, scale=1.0):
		'''Initializes the InitialOrbitDetermination class

		Attributes:
			noradID - NORAD ID of object
			detectionID - Detection Request ID of object pass
			method - Which IOD method to use
			scale - 0.0 to 1.0 for what percentage of pass to use
		'''
		#Define Initializing Variables
		self.noradID = noradID
		self.detectionRequestID = detectionRequestID
		self.method = method
		self.scale = scale

		#NORAD ID and Detection Request ID
		initialPass = Pass(self.noradID, self.detectionRequestID)
		self.passLength = self.scale*initialPass.getPassLength()

		#Process Dataframe into lists for initial orbit determination
		self.az, self.el,  \
		self.ra, self.dec, \
		self.time, self.slantRange, \
		self.rSite = initialPass.getThreePointCentered(self.passLength/2)

		#Process angle rates for initial orbit determination
		self.azRate, self.elRate, \
		self.raRate, self.decRate = initialPass.threePointCenteredAngleRates(self.passLength/2)

		#Evalute Chosen IOD Method
		self.initialOrbit = self.evaluateMethod()

	def evaluateMethod(self):
		'''Evaluates the chosen IOD method

		Returns:
			initialOrbit - orekit orbit object
		'''
		#Determine which IOD method to use
		if self.method == 'doubleR':      
			#Double-R Method
			initialOrbit, rho1, rho2, rho3, r2vec, v2vec = self.doubleRMethod()

		elif self.method == 'gauss':
			#Gauss Method with Differential Correction
			initialOrbit, rho1, rho2, rho3, r2vec, v2vec = self.gaussMethod()

		elif self.method == 'gooding':
			#Double-R Method for preliminary guess for range values
			initialOrbit, rho1, rho2, rho3, r2vec, v2vec = self.gaussMethod()

			#Gooding Initial Orbit Determination for better estimate
			iod = IodGooding(ECI, EARTH_MU)
			initialOrbit = iod.estimate(self.rSite[0],self.rSite[1],self.rSite[2],
					self.slantRange[0],datetime_to_absolutedate(self.time[0]),
					self.slantRange[1],datetime_to_absolutedate(self.time[1]),
					self.slantRange[2],datetime_to_absolutedate(self.time[2]),
					float(rho1*KM2M),float(rho3*KM2M))

			#Define which velocity method was used
			self.velocityMethod = 'gooding'

		elif self.method == 'modified':
			#Combination of Gooding and Gauss
			initialOrbit = self.modifiedMethod()

			#Define which velocity method was used
			self.velocityMethod = 'modified'

		elif self.method == 'circular':
			#Circular Orbit Assumption
			initialOrbit, rho1, rho2, rho3, r2vec, v2vec = self.assumedCircularOrbitMethod()

			#Define which velocity method was used
			self.velocityMethod = 'circular'

		else:
			#Invalud Method
			error('Invalid IOD Method: (doubleR // gauss // gooding // modified')

		return initialOrbit

	def modifiedMethod(self):
		'''Calculates a position and velocity vector in earth
		centered inertial for the middle observation.

		Returns:
			r2vec - position vector for second measurement
			v2vec - velocity vector for second measurement
		'''
		#Gauss Method for Position Vector
		orbitGauss, _, _, _, _, _ = self.gaussMethod()
		pvGauss = orbitGauss.getPVCoordinates()

		#Gooding + Double-R Method for Velocity Vector
		_, rho1, rho2, rho3, _, _ = self.gaussMethod()

		#Gooding Initial Orbit Determination for better estimate
		iod = IodGooding(ECI, EARTH_MU)
		orbitGooding = iod.estimate(self.rSite[0],self.rSite[1],self.rSite[2],
				self.slantRange[0],datetime_to_absolutedate(self.time[0]),
				self.slantRange[1],datetime_to_absolutedate(self.time[1]),
				self.slantRange[2],datetime_to_absolutedate(self.time[2]),
				float(rho1*KM2M),float(rho3*KM2M))
		pvGooding = orbitGooding.getPVCoordinates()

		#Get Position and Velocity Vectors from respective methods
		position = pvGauss.getPosition()
		velocity = pvGooding.getVelocity()

		#Generate Orbit
		pv = TimeStampedPVCoordinates(datetime_to_absolutedate(self.time[1]),
				position,velocity)
		initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

		return initialOrbit

	def doubleRMethod(self):
		'''Calculates a position and velocity vector in earth
		centered inertial for the middle observation.

		Returns:
			r2vec - position vector for second measurement
			v2vec - velocity vector for second measurement
		'''
		#Format Slant Range Vectors
		rho1 = [self.slantRange[0].getX(),self.slantRange[0].getY(),self.slantRange[0].getZ()]
		rho2 = [self.slantRange[1].getX(),self.slantRange[1].getY(),self.slantRange[1].getZ()]
		rho3 = [self.slantRange[2].getX(),self.slantRange[2].getY(),self.slantRange[2].getZ()]

		#Format Site Vectors
		R1 = np.multiply([self.rSite[0].getX(),self.rSite[0].getY(),self.rSite[0].getZ()],M2KM)
		R2 = np.multiply([self.rSite[1].getX(),self.rSite[1].getY(),self.rSite[1].getZ()],M2KM)
		R3 = np.multiply([self.rSite[2].getX(),self.rSite[2].getY(),self.rSite[2].getZ()],M2KM)

		#Format Time Separation
		tau1 = (self.time[0] - self.time[1]).total_seconds()
		tau3 = (self.time[2] - self.time[1]).total_seconds()

		#Initial guesses for iteration
		r1 = 1.5*EARTH_RADIUS_KM
		r2 = 1.51*EARTH_RADIUS_KM
		c1 = np.dot(np.multiply(2,rho1),R1)
		c2 = np.dot(np.multiply(2,rho2),R2)

		#Iteration Paramters
		error = 1
		tol = 1e-5
		n = 0
		nmax = 100

		#Iteratively solve until error reaches tolerance
		while (error > tol) and (n < nmax):
			#Increase counter
			n += 1

			#Modify position magnitudes for differential correction
			F1,F2,f,g,r1vec,r2vec,r3vec,q1,q2,q3 = self.dr_function(c1,c2,r1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)
			dr1 = np.multiply(0.005,r1)
			dr2 = np.multiply(0.005,r2)
			F1r1dr1,F2r1dr1,_,_,_,_,_,_,_,_ = self.dr_function(c1,c2,r1+dr1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)
			F1r2dr2,F2r2dr2,_,_,_,_,_,_,_,_ = self.dr_function(c1,c2,r1,r2+dr2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3)

			#Calculate differences
			dF1dr1 = (F1r1dr1 - F1)/dr1
			dF2dr1 = (F2r1dr1 - F2)/dr1
			dF1dr2 = (F1r2dr2 - F1)/dr2
			dF2dr2 = (F2r2dr2 - F2)/dr2
			
			#Calculate recommended magnitude changes
			delta = dF1dr1*dF2dr2 - dF2dr1*dF1dr2
			delta1 = dF2dr2*F1 - dF1dr2*F2
			delta2 = dF1dr1*F2 - dF2dr1*F1
			dr1 = -delta1/delta
			dr2 = -delta2/delta

			#Error calculation
			error = (np.absolute(dr1) + np.absolute(dr2))/2;

			#Update Magnitude of position vectors
			r1 = r1 + dr1
			r2 = r2 + dr2

		#Angular Separation between Variables
		self.alpha12, self.alpha23, self.alpha13 = self.arcLength(r1vec,r2vec,r3vec)

		#Using Position Vectors to get best Velocity Estimate
		v2vec, _ = self.lambertsProblem(r2vec,r3vec,tau3,-1)

		#Create Orekit Orbit object
		pv = TimeStampedPVCoordinates(datetime_to_absolutedate(self.time[1]),
				Vector3D(float(r2vec[0]*KM2M),float(r2vec[1]*KM2M),float(r2vec[2]*KM2M)),
				Vector3D(float(v2vec[0]*KM2M),float(v2vec[1]*KM2M),float(v2vec[2]*KM2M)))
		initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

		return initialOrbit, q1, q2, q3, r2vec, v2vec		

	def gaussMethod(self):
		'''Calculates a position and velocity vector in earth
		centered inertial for the middle observation.

		Returns:
			r2vec - position vector for second measurement
			v2vec - velocity vector for second measurement
		'''
		#Format Slant Range Vectors
		rho1 = np.array([self.slantRange[0].getX(),self.slantRange[0].getY(),self.slantRange[0].getZ()])
		rho2 = np.array([self.slantRange[1].getX(),self.slantRange[1].getY(),self.slantRange[1].getZ()])
		rho3 = np.array([self.slantRange[2].getX(),self.slantRange[2].getY(),self.slantRange[2].getZ()])

		#Format Site Vectors
		R1 = np.multiply([self.rSite[0].getX(),self.rSite[0].getY(),self.rSite[0].getZ()],M2KM)
		R2 = np.multiply([self.rSite[1].getX(),self.rSite[1].getY(),self.rSite[1].getZ()],M2KM)
		R3 = np.multiply([self.rSite[2].getX(),self.rSite[2].getY(),self.rSite[2].getZ()],M2KM)

		#Format Time Separation
		tau1 = (self.time[0] - self.time[1]).total_seconds()
		tau3 = (self.time[2] - self.time[1]).total_seconds()
		
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

		#Velocity vector
		v2vec = (-f3*r1vec + f1*r3vec)/(f1*g3 - f3*g1)

		#Save first iteration variables
		q1_old = q1
		q2_old = q2
		q3_old = q3

		#Iteration set up variables
		err = 1
		n = 0
		nmax = 100
		tol = 1e-5

		#Iterative improvement of gauss
		while (err > tol) and (n < nmax):
			#Increase Counter
			n += 1

			#Compute universal kepler's equation parameters
			ro = norm(r2vec)
			vo = norm(v2vec)
			vro = np.dot(v2vec,r2vec)/ro
			a = (2/ro) - (vo**2)/EARTH_MU_KM

			#Kepler equations
			_, _, ff1, gg1, _, _, _ = self.universalVariables(r2vec,v2vec,tau1)
			_, _, ff3, gg3, _, _, _ = self.universalVariables(r2vec,v2vec,tau3)
			
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
			v2vec = (-f3*r1vec + f1*r3vec)/(f1*g3 - f3*g1)

			#Error
			err = np.sqrt((q1 - q1_old)**2 + (q2 - q2_old)**2 + (q3 - q3_old)**2)

			#Update slant ranges
			q1_old = q1
			q2_old = q2
			q3_old = q3

		#Angular Separation between Variables
		self.alpha12, self.alpha23, self.alpha13 = self.arcLength(r1vec,r2vec,r3vec)

		#Using Position Vectors to get best Velocity Estimate
		v2vec = self.velocityCorrectionMethods(r1vec,r2vec,r3vec)

		#Create Orekit Orbit object
		pv = TimeStampedPVCoordinates(datetime_to_absolutedate(self.time[1]),
				Vector3D(float(r2vec[0]*KM2M),float(r2vec[1]*KM2M),float(r2vec[2]*KM2M)),
				Vector3D(float(v2vec[0]*KM2M),float(v2vec[1]*KM2M),float(v2vec[2]*KM2M)))
		initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

		return initialOrbit, q1, q2, q3, r2vec, v2vec

	def assumedCircularOrbitMethod(self):
		'''Calculates a position and velocity vector in earth
		centered inertial for the middle observation.

		Returns:
			r2vec - position vector for second measurement
			v2vec - velocity vector for second measurement
		'''
		#Format Slant Range Vectors
		rho1 = np.array([self.slantRange[0].getX(),self.slantRange[0].getY(),self.slantRange[0].getZ()])
		rho2 = np.array([self.slantRange[1].getX(),self.slantRange[1].getY(),self.slantRange[1].getZ()])
		rho3 = np.array([self.slantRange[2].getX(),self.slantRange[2].getY(),self.slantRange[2].getZ()])

		#Format Site Vectors
		R1 = np.multiply([self.rSite[0].getX(),self.rSite[0].getY(),self.rSite[0].getZ()],M2KM)
		R2 = np.multiply([self.rSite[1].getX(),self.rSite[1].getY(),self.rSite[1].getZ()],M2KM)
		R3 = np.multiply([self.rSite[2].getX(),self.rSite[2].getY(),self.rSite[2].getZ()],M2KM)

		#Format Time Separation
		tau1 = (self.time[0] - self.time[1]).total_seconds()
		tau3 = (self.time[2] - self.time[1]).total_seconds()
		
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

		#Velocity vector from Lagrange Equations
		v2vec = (-f3*r1vec + f1*r3vec)/(f1*g3 - f3*g1)

		#Save first iteration variables
		q1_old = q1
		q2_old = q2
		q3_old = q3

		#Iteration set up variables
		err = 1
		n = 0
		nmax = 100
		tol = 1e-5

		#Iterative improvement of gauss
		while (err > tol) and (n < nmax):
			#Increase Counter
			n += 1

			#Compute universal kepler's equation parameters
			ro = norm(r2vec)
			vo = norm(v2vec)
			vro = np.dot(v2vec,r2vec)/ro
			a = (2/ro) - (vo**2)/EARTH_MU_KM

			#Kepler equations
			_, _, ff1, gg1, _, _, _ = self.universalVariables(r2vec,v2vec,tau1)
			_, _, ff3, gg3, _, _, _ = self.universalVariables(r2vec,v2vec,tau3)
			
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
			v2vec = (-f3*r1vec + f1*r3vec)/(f1*g3 - f3*g1)

			#Error
			err = np.sqrt((q1 - q1_old)**2 + (q2 - q2_old)**2 + (q3 - q3_old)**2)

			#Update slant ranges
			q1_old = q1
			q2_old = q2
			q3_old = q3

		#Angular Separation between Variables
		self.alpha12, self.alpha23, self.alpha13 = self.arcLength(r1vec,r2vec,r3vec)

		#Assume Circular Orbit
		r2 = np.linalg.norm(r2vec)
		h = np.sqrt(r2*EARTH_MU_KM)
		vMag = h/r2
		h2vec = np.cross(r2vec,v2vec)
		v2unit = np.cross(np.divide(h2vec,np.linalg.norm(h2vec)),np.divide(r2vec,np.linalg.norm(r2vec)))
		v2vec = np.multiply(v2unit,vMag)

		#Create Orekit Orbit object
		pv = TimeStampedPVCoordinates(datetime_to_absolutedate(self.time[1]),
				Vector3D(float(r2vec[0]*KM2M),float(r2vec[1]*KM2M),float(r2vec[2]*KM2M)),
				Vector3D(float(v2vec[0]*KM2M),float(v2vec[1]*KM2M),float(v2vec[2]*KM2M)))
		initialOrbit = KeplerianOrbit(pv, ECI, EARTH_MU)

		return initialOrbit, q1, q2, q3, r2vec, v2vec

	def dr_function(self,c1,c2,r1,r2,tau1,tau3,rho1,rho2,rho3,R1,R2,R3):
		'''Sub-function that calculates lagrangian multipliers 
		as well as other function values for the main iteration.

		Attributes:
			c - angle constants
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
		q2 = (-c2 + np.sqrt((c2**2) - 4*(np.dot(R2,R2) - (r2**2))))/2

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
		if wvec[2] > 0:
			sin21 = np.sqrt(1-(cos21**2))
			sin31 = np.sqrt(1-(cos31**2))
			sin32 = np.sqrt(1-(cos31**2))
		else:
			sin21 = -np.sqrt(1-(cos21**2))
			sin31 = -np.sqrt(1-(cos31**2))
			sin32 = -np.sqrt(1-(cos31**2))

		#Calculate other constants   
		theta31 = np.arccos(cos31)
		if theta31 > pi/2:
			c1 = (r2*sin32)/(r1*sin31)
			c3 = (r2*sin21)/(r3*sin31)
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

		#Ellipctical Case
		if np.sqrt(e2) < 1:
			n = np.sqrt(EARTH_MU_KM/(a**3))
			S = r2/p*np.sqrt(1-e2)*esin2
			C = r2/p*(e2 + ecos2)
			sinE32 = r3/np.sqrt(a*p)*sin32 - r3/p*(1 - cos32)*S
			cosE32 = 1 - (r2*r3)/(a*p)*(1-cos32)
			sinE21 = r1/np.sqrt(a*p)*sin21 + r1/p*(1 - cos21)*S
			cosE21 = 1 - (r2*r1)/(a*p)*(1-cos21)
			M32 = np.arccos(cosE32) + 2*S*np.sin(np.arccos(cosE32)/2)**2 - C*sinE32
			M12 = -np.arccos(cosE21) + 2*S*np.sin(np.arccos(cosE21)/2)**2 + C*sinE21
			F1 = tau1 - M12/n
			F2 = tau3 - M32/n
			f = 1 - a/r2*(1 - cosE32)
			g = tau3 - np.sqrt((a**3)/EARTH_MU_KM)*(np.arccos(cosE32) - sinE32)
		#Hyperbolic Case
		else:
			n = np.sqrt(EARTH_MU_KM/(-a)**3)
			Sh = r2/p*np.sqrt(e2-1)*esin2
			Ch = r2/p*(e2 + ecos2)
			sinhF32 = r3/np.sqrt(-a*p)*sin32 - r3/p*(1 - cos32)*Sh
			F32 = np.log(sinhF32 + np.sqrt(sinhF32 + 1))
			sinhF21 = r1/np.sqrt(-a*p)*sin21 + r1/p*(1 - cos32)*Sh
			F21 = np.log(sinhF21 + np.sqrt(sinhF21**2 + 1))
			M32 = -F32 + 2*Sh*np.sinh(F32/2)**2 + Ch*sinhF32
			M12 = F21 + 2*Sh*np.sinh(F21/2)**2 + Ch*sinhF21
			F1 = tau1 - M12/n
			F2 = tau3 - M32/n
			f = 1 - (-a)/r2*(1 - np.cosh(F32))
			g = tau3 - np.sqrt(((-a)**3)/EARTH_MU_KM)*(F32 - sinhF32)

		return F1, F2, f, g, r1vec, r2vec, r3vec, q1, q2, q3

	def velocityCorrectionMethods(self,r1vec,r2vec,r3vec):
		'''Uses Lambert's Solution, Gibbs or Herrick-Gibbs Methods 
		for calculating the velocity vector of middle position from 
		three position vectors depending on different conditions.

		Attributes:
			r1vec - position vector at time 1
			r2vec - position vector at time 2
			r3vec - position vector at time 3
		
		Returns:
			v2vec - velocity vector at time 2
		'''
		#Norms of the Position Vectors
		r1 = norm(r1vec)
		r2 = norm(r2vec)
		r3 = norm(r3vec)

		#Time Separation between measurements
		dt31 = ((self.time[2] - self.time[0]).total_seconds())
		dt32 = ((self.time[2] - self.time[1]).total_seconds())
		dt21 = ((self.time[1] - self.time[0]).total_seconds())

		#Position Vector Cross Products
		Z12 = np.cross(r1vec,r2vec)
		Z23 = np.cross(r2vec,r3vec)
		Z31 = np.cross(r3vec,r1vec)

		#Coplanar Check
		alphaCOP = 90 - degrees(np.arccos(np.dot(Z23,r1vec)/(norm(Z23)*r1)))

		#Check Angular Separation between Vectors
		alpha12, alpha23, alpha13 = self.arcLength(r1vec,r2vec,r3vec)

		#Check if Vectors are Co-Planer
		if alphaCOP > 1:
			#Not Coplanar, use Lambert's Solution
			v2vec, _ = self.lambertsProblem(r2vec,r3vec,dt32,-1)

			#Define which velocity method was used
			self.velocityMethod = 'lambert'
		else:
			#Check to use Gibbs or Herrick-Gibbs
			if (alpha12 < 1) or (alpha23 < 1):
				#Herrick-Gibbs Taylor Series Approximation
				A = -dt32*(1/(dt21*dt31) + EARTH_MU_KM/(12*(r1**3)))
				B = (dt32 - dt21)*(1/(dt21*dt32) + EARTH_MU_KM/(12*(r2**3)))
				C = dt21*(1/(dt32*dt31) + EARTH_MU_KM/(12*(r2**3)))

				#Velocity Vector
				v2vec = np.multiply(A,r1vec) + np.multiply(B,r2vec) + np.multiply(C,r3vec)

				#Define which velocity method was used
				self.velocityMethod = 'Herrick-Gibbs'
			else:
				#Gibbs Geometrical Calculation
				N = np.multiply(r1,Z23) + np.multiply(r2,Z31) + np.multiply(r3,Z12) 
				D = np.add(np.add(Z12,Z23),Z31)
				S = np.multiply(r2 - r3,r1vec) + np.multiply(r3 - r1,r2vec) + np.multiply(r1 - r2,r3vec)
				B = np.cross(D,r2vec)
				L = np.sqrt(EARTH_MU_KM/np.dot(N,D))

				#Velocity Vector
				v2vec = np.multiply(L/r2,B) + np.multiply(L,S)

				#Define which velocity method was used
				self.velocityMethod = 'gibbs'

		return v2vec

	def arcLength(self,r1vec,r2vec,r3vec):

		#Norms of the Position Vectors
		r1 = norm(r1vec)
		r2 = norm(r2vec)
		r3 = norm(r3vec)

		#Check Angular Separation between Vectors
		alpha12 = degrees(np.arccos(np.dot(r1vec,r2vec)/(r1*r2)))
		alpha23 = degrees(np.arccos(np.dot(r2vec,r3vec)/(r2*r3)))
		alpha13 = degrees(np.arccos(np.dot(r1vec,r3vec)/(r1*r3)))	

		return alpha12, alpha23, alpha13

	def lambertsProblem(self,r1vec,r2vec,dt,path):
		'''Solves the lambert problem to determine the velocity
		vector from two position vectors.

		Source: 
		Shiva Iyer (2020). Solve Lambert's Problem in Two-Body Dynamics 
		(https://www.mathworks.com/matlabcentral/fileexchange/44789-solve-lambert-s-problem-in-two-body-dynamics), 
		MATLAB Central File Exchange. Retrieved May 20, 2020.

		Attributes:
			r1vec - position vector at time 1
			r2vec - position vector at time 2
			dt - propagation time
			path - determines which direction: short way(-1) // long way(1)

		Returns:
			v1vec - velocity vector at time 1
			v2vec - velocity vector at time 2
		'''
		#Position Vectors
		r1norm = np.linalg.norm(r1vec)
		r2norm = np.linalg.norm(r2vec)

		#Lambert Solution Variables
		mutt = np.sqrt(EARTH_MU_KM)*dt
		dnu = np.arccos(np.dot(r1vec, r2vec)/(r1norm*r2norm))
		A = np.sqrt(r1norm*r2norm*(1+np.cos(dnu)))
		if path == 1:
			A = -A

		#Iteratively Solve for Z
		z = 0
		for i in range(0,50):
			#Stumpff Functions
			S = self.stumpS(z)
			C = self.stumpC(z)
			y = np.absolute(r1norm+r2norm - A*(1-z*S)/np.sqrt(C))
			x = np.sqrt(y/C)
			t = (x**3)*S + A*np.sqrt(y)

			#Calculating time
			if np.absolute(z) > 1e-4:
				Cp = (1-z*S-2*C)/(2*z)
				Sp = (C-3*S)/(2*z)
				tp = (x**3)*(Sp-1.5*S*Cp/C)+0.125*A*(3*S*np.sqrt(y)/C+A/x)
			else:
				tp = (np.sqrt(2)/40)*(y**1.5)+0.125*A*(np.sqrt(y)+A*np.sqrt(0.5/y))

			#Convergence Conditions
			#if np.absolute(t - mutt) < 1e-4:
			#Lagrange Multipliers
			f = 1-y/r1norm
			g = A*np.sqrt(y/EARTH_MU_KM)
			gd = 1 -y/r2norm

			#Velocity Vectors
			v1vec = (r2vec - f*r1vec)/g
			v2vec = (gd*r2vec - r1vec)/g
			#	break

			#Calculate Z
			z = z-(t-mutt)/tp

		return v1vec, v2vec

	def universalVariables(self,r2vec,v2vec,dt):
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
		a = (2/ro) - (vo**2)/EARTH_MU_KM
		
		#Initial universal variable
		X = np.sqrt(EARTH_MU_KM)*np.absolute(a)*dt

		#Set error tolerance
		tol = 1e-8
		nmax = 5000

		#Newton's method for iteration
		ratio = 1
		n = 0
		count = 1

		#Iteratively solve for X and z
		while (np.absolute(ratio) > tol) and (n < nmax):
			n += 1
			z = a*(X**2)
			funcX = ((ro*vro)/np.sqrt(EARTH_MU_KM))*(X**2)*self.stumpC(z) + (1-a*ro)*(X**3)*self.stumpS(z) + ro*X-np.sqrt(EARTH_MU_KM)*dt
			funcXdot = ((ro*vro)/np.sqrt(EARTH_MU_KM))*X*(1-a*(X**2)*self.stumpS(z)) + (1-a*ro)*(X**2)*self.stumpC(z)+ro
			ratio = funcX/funcXdot
			X = X - ratio
			count += 1
		z = a*(X**2)

		#Lagrange multipliers
		f = 1 - ((X**2)/ro)*self.stumpC(z)
		g = dt - (1/np.sqrt(EARTH_MU_KM))*(X**3)*self.stumpS(z)

		#Position Vector
		rvec = f*r2vec + g*v2vec
		r = norm(rvec)

		#Lagrange multipliers
		fdot = (np.sqrt(EARTH_MU_KM)/(r*ro))*(a*(X**3)*self.stumpS(z) - X)
		gdot = 1 - ((X**2)/r)*self.stumpC(z)

		#Velocity Vector
		vvec = fdot*r2vec + gdot*v2vec

		return rvec, vvec, f, g, fdot, gdot, X

	def stumpS(self,z):
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

	def stumpC(self,z):
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
