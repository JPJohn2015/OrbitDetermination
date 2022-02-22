'''
Main script for Orbit Determination.

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
from org.orekit.utils import IERSConventions, Constants, CartesianDerivativesFilter
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.propagation.analytical.tle import TLE
from org.orekit.attitudes import NadirPointing
from org.orekit.propagation.analytical.tle import SGP4
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder
from org.orekit.orbits import PositionAngle
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.forces.gravity import Relativity
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.estimation.iod import IodGooding
from org.orekit.estimation.measurements import AngularAzEl, GroundStation, ObservableSatellite
from org.hipparchus.linear import Array2DRowRealMatrix
from org.hipparchus.geometry.euclidean.threed import Vector3D, Rotation
from orekit.pyhelpers import *
from orekit import JArray

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

########## DATA MANAGER CONFIGURATION

#Data Manager
orekit_filename = 'orekit-data'
DM = DataProvidersManager.getInstance()
datafile = File(orekit_filename)
if not datafile.exists():
    print('Directory :', datafile.absolutePath, ' not found')
crawler = DirectoryCrawler(datafile)
DM.clearProviders()
DM.addProvider(crawler)

########## COORDINATE FRAMES, CELESTIAL BODIES AND TIME

#Ground Station Position
latitude = 37.1348
longitude = -122.2110
altitude = 684.0

#Defining ECI AND ECEF Coordinate Frames
gcrf = FramesFactory.getGCRF()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
eci = gcrf
ecef = itrf

#Defining Topocentric Coordinate Frame
tod = FramesFactory.getTOD(IERSConventions.IERS_2010, False)
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                         Constants.WGS84_EARTH_FLATTENING, 
                         itrf)
station = GeodeticPoint(radians(latitude), 
                        radians(longitude), 
                        altitude)
stationFrame = TopocentricFrame(earth, station, "Esrange")

#Defining Celestial Bodies
wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(ecef)
moon = CelestialBodyFactory.getMoon()
sun = CelestialBodyFactory.getSun()

#Defining Time System
utc = TimeScalesFactory.getUTC()

########## INITIAL ORBIT DETERMINATION

print("Initial Orbit Determination")

#Satellite NORAD ID being observed
#noradID = 877
noradID = 40380

#Altitude Guess in km
orbitHeight = [100,900]

#Search Pass Log for NORAD ID and get list of Detection Request IDs
passLog = loadPassLog()
noradIDPassLog = passLog.loc[(passLog['noradID'] == noradID) & (passLog['passLength'] > 10)]
detectionRequestID = list(noradIDPassLog['detectionRequestID'])
print(detectionRequestID)

#First NORAD ID and Detection Request ID
orbitalPass = Pass(noradID, detectionRequestID[0])

#Process Dataframe into lists for initial orbit determination
az, el, ra, dec, datetime, slantRange = orbitalPass.getThreePointCentered(orbitalPass.getPassLength()/2)

#Calculate position vector for ground station
rSiteList = [stationSite(latitude,longitude,altitude,datetime[0]),
             stationSite(latitude,longitude,altitude,datetime[1]),
             stationSite(latitude,longitude,altitude,datetime[2])]

#Initialize List Variables
inclination = []
eccentricity = []
semiMajorAxis = []
parameter = []

#Loop through orbital heights to find minimum of Eccentricity
for i in range(1,orbitHeight[1]):
    #Gooding Initial Orbit Determination
    rho1, rho3 = RangeGuess(el[0],el[2],(orbitHeight[0]+i))
    iod = IodGooding(eci, Constants.WGS84_EARTH_MU)
    keplerianOrbit = iod.estimate(rSiteList[0],rSiteList[1],rSiteList[2],
            slantRange[0],datetime_to_absolutedate(datetime[0]),
            slantRange[1],datetime_to_absolutedate(datetime[1]),
            slantRange[2],datetime_to_absolutedate(datetime[2]),
            rho1,rho3)

    #Append Values to Lists
    parameter.append(i)
    inclination.append(degrees(keplerianOrbit.getI()))
    eccentricity.append(keplerianOrbit.getE())
    semiMajorAxis.append(keplerianOrbit.getA())

#Index of minimum of eccentricity
idx = np.argmin(eccentricity)

#Best Range Guess
rho1, rho3 = RangeGuess(el[0],el[2],(orbitHeight[0]+idx))

#Gooding Initial Orbit Determination
iod = IodGooding(eci, Constants.WGS84_EARTH_MU)
keplerianOrbit = iod.estimate(rSiteList[0],rSiteList[1],rSiteList[2],
            slantRange[0],datetime_to_absolutedate(datetime[0]),
            slantRange[1],datetime_to_absolutedate(datetime[1]),
            slantRange[2],datetime_to_absolutedate(datetime[2]),
            rho1,rho3)

#Plotting Iterative Gooding Method to check for Convergence

'''
plt.figure
plt.subplot(1,3,1)
plt.plot(parameter,semiMajorAxis)
plt.subplot(1,3,2)
plt.plot(parameter,eccentricity)
plt.subplot(1,3,3)
plt.plot(parameter,inclination)
plt.show()
'''

'''
print("Initial Orbit Determination Parameters")
print("Eccentricity = " + str(keplerianOrbit.getE()))
print("Semi Major Axis = " + str(keplerianOrbit.getA()))
print("Inclination = " + str(degrees(keplerianOrbit.getI())))
print("Right Ascension = " + str(degrees(keplerianOrbit.getRightAscensionOfAscendingNode())))
print("Argument of Perigee = " + str(degrees(keplerianOrbit.getPerigeeArgument())))
print("True Anomaly = " + str(degrees(keplerianOrbit.getTrueAnomaly())))

print("-------")
print(keplerianOrbit.getPVCoordinates())
'''
input("Press ENTER to continue")

########## PROPAGATOR SETUP

print("Starting Propagator Setup")

#Orbit Propagator Parameters
prop_min_step = 0.001 
prop_max_step = 300.0 
prop_position_error = 10.0 

#Estimator Parameters
estimator_position_scale = 1.0 
estimator_convergence_thres = 1e-3
estimator_max_iterations = 5
estimator_max_evaluations = 35

#Measurement Parameters
azError = 0.001
elError = 0.001
azBaseWeight = 1.0
elBaseWeight = 1.0

#Defining Integrator
integratorBuilder = DormandPrince853IntegratorBuilder(prop_min_step, prop_max_step, prop_position_error)
propagatorBuilder = NumericalPropagatorBuilder(keplerianOrbit,
                                               integratorBuilder, PositionAngle.MEAN, estimator_position_scale)
nadirPointing = NadirPointing(eci, wgs84Ellipsoid)
propagatorBuilder.setMass(400.0)
propagatorBuilder.setAttitudeProvider(nadirPointing)

########## ADDING PERTURBATIONS

#Earth Gravity Field
gravityProvider = GravityFieldFactory.getConstantNormalizedProvider(64, 64)
gravityAttractionModel = HolmesFeatherstoneAttractionModel(ecef, gravityProvider)
propagatorBuilder.addForceModel(gravityAttractionModel)

#3rd Body
moon_3dbodyattraction = ThirdBodyAttraction(moon)
propagatorBuilder.addForceModel(moon_3dbodyattraction)
sun_3dbodyattraction = ThirdBodyAttraction(sun)
propagatorBuilder.addForceModel(sun_3dbodyattraction)

'''
#Atmospheric Drag
msafe = MarshallSolarActivityFutureEstimation(
    '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\p{Digit}\p{Digit}\p{Digit}\p{Digit}F10\.(?:txt|TXT)',
    MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
DM.feed(msafe.getSupportedNames(), msafe) # Feeding the F10.7 bulletins to Orekit's data manager
atmosphere = NRLMSISE00(msafe, sun, wgs84Ellipsoid)
isotropicDrag = IsotropicDrag(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cd'])
dragForce = DragForce(atmosphere, isotropicDrag)
propagatorBuilder.addForceModel(dragForce)

# Solar radiation pressure
isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cr']);
solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid.getEquatorialRadius(),
                                                isotropicRadiationSingleCoeff)
propagatorBuilder.addForceModel(solarRadiationPressure)
'''

########## BATCH LEAST SQUARES ESTIMATOR SETUP

#Defining Optimizer
matrixDecomposer = QRDecomposer(1e-11)
optimizer = GaussNewtonOptimizer(matrixDecomposer, False)

#Defining Estimator
estimator = BatchLSEstimator(optimizer, propagatorBuilder)
estimator.setParametersConvergenceThreshold(estimator_convergence_thres)
estimator.setMaxIterations(estimator_max_iterations)
estimator.setMaxEvaluations(estimator_max_evaluations)

########## ADDING MEASUREMENTS

print("Adding Measurements")

for i in range(1,2):#len(detectionRequestID)):
    #First NORAD ID and Detection Request ID
    orbitalPass = Pass(noradID, detectionRequestID[0])

    #Process Dataframe into lists for initial orbit determination
    az, el, ra, dec, datetime, slantRange = orbitalPass.getConstantSpacing(1)

    #Add Measurements to Estimator
    for j in range(1,len(az)):
        orekitAzEl = AngularAzEl(GroundStation(stationFrame),
    					datetime_to_absolutedate(datetime[j]),
    					JArray('double')([radians(az[j]),radians(el[j])]),
                        JArray('double')([radians(azError),radians(elError)]),
                        JArray('double')([azBaseWeight,elBaseWeight]),
                        ObservableSatellite(0))
        estimator.addMeasurement(orekitAzEl)

########## STATISTICAL ORBIT DETERMINATION

print("Estimating State")

estimatedPropagatorArray = estimator.estimate()

estimatedPropagator = estimatedPropagatorArray[0]
estimatedInitialState = estimatedPropagator.getInitialState()
actualOdDate = estimatedInitialState.getDate()
estimatedOrbit_init = estimatedInitialState.getOrbit()

########## COVARIANCE ANALYSIS

# Creating the LVLH frame 
lvlh = LocalOrbitalFrame(eci, LOFType.LVLH, estimatedPropagator, 'LVLH')

# Getting covariance matrix in ECI frame
covMat_eci_java = estimator.getPhysicalCovariances(1.0e-10)

# Converting matrix to LVLH frame
# Getting an inertial frame aligned with the LVLH frame at this instant
# The LVLH is normally not inertial, but this should not affect results too much
# Reference: David Vallado, Covariance Transformations for Satellite Flight Dynamics Operations, 2003
eci2lvlh_frozen = eci.getTransformTo(lvlh, actualOdDate).freeze() 

# Computing Jacobian
jacobianDoubleArray = JArray_double2D(6, 6)
eci2lvlh_frozen.getJacobian(CartesianDerivativesFilter.USE_PV, jacobianDoubleArray)
jacobian = Array2DRowRealMatrix(jacobianDoubleArray)
# Applying Jacobian to convert matrix to lvlh
covMat_lvlh_java = jacobian.multiply(
    covMat_eci_java.multiply(covMat_eci_java.transpose()))

# Converting the Java matrices to numpy
covarianceMat_eci = np.matrix([covMat_eci_java.getRow(iRow) 
                              for iRow in range(0, covMat_eci_java.getRowDimension())])
covarianceMat_lvlh = np.matrix([covMat_lvlh_java.getRow(iRow) 
                              for iRow in range(0, covMat_lvlh_java.getRowDimension())])

# Computing the position and velocity standard deviation
pos_std_crossTrack = np.sqrt(max(0.0, covarianceMat_lvlh[0,0]))
pos_std_alongTrack = np.sqrt(max(0.0, covarianceMat_lvlh[1,1]))
pos_std_outOfPlane = np.sqrt(max(0.0, covarianceMat_lvlh[2,2]))
vel_std_crossTrack = np.sqrt(max(0.0, covarianceMat_lvlh[3,3])) # In case the value is negative...
vel_std_alongTrack = np.sqrt(max(0.0, covarianceMat_lvlh[4,4]))
vel_std_outOfPlane = np.sqrt(max(0.0, covarianceMat_lvlh[5,5]))



'''
#Kyle's Test Case
rightAscension = [5.510755882602251, 5.649241527922573, 5.812326035706719]
declination = [0.185610983747447, 0.359937558669416, 0.527401708412054]
elevation = [degrees(1.042670760528870), degrees(1.106073650425334), degrees(1.075063460580647)]
julianDate = [2.458417575892785*1e6, 2.458417576182427*1e6, 2.458417576472068*1e6]

slantRange = []
datetime = []
for i in range(0,len(rightAscension)):
    ra = rightAscension[i]
    dec = declination[i]
    los = Vector3D(float(cos(dec)*cos(ra)),
                    float(cos(dec)*sin(ra)),
                    float(sin(dec)))
    slantRange.append(los)
    datetime.append(from_jd(julianDate[i]))
'''


