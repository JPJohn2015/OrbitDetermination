'''
Main script for Orbit Determination.

Authors:    James Johnson

Created:    Feb-2020
'''

#Initialize OREkit and Java Virtual Machine
import orekit
vm = orekit.initVM()

#Import Orekit Libraries
from org.orekit.utils import IERSConventions, Constants
from org.orekit.frames import FramesFactory, ITRFVersion, TopocentricFrame, LocalOrbitalFrame, LOFType, Transform
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory

#Import Python Libraries
from datetime import datetime, timedelta
from math import *



########## DATA PROCESSING VARIABLES
START_TIME = datetime.now()
'''Datetime at beginning of Script for Status Tracking'''



########## EARTH VARIABLES
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
'''WGS 84 Model for Earth Equatorial Radius [m]'''

EARTH_RADIUS_KM = Constants.WGS84_EARTH_EQUATORIAL_RADIUS*(1e-3)
'''WGS 84 Model for Earth Equatorial Radius [km]'''

EARTH_MU = Constants.WGS84_EARTH_MU
'''WGS 84 Model for Earth Gravitational Constant [m^3/s^2]'''

EARTH_MU_KM = Constants.WGS84_EARTH_MU*(1e-9)
'''WGS 84 Model for Earth Gravitational Constant [km^3/s^2]'''

EARTH_FLATTENING = Constants.WGS84_EARTH_FLATTENING
'''WGS 84 Model for Earth Flattening Factor'''

DENSITY = 1.225
'''Atmospheric Density at sea level [kg/m^3]'''

HEIGHT_SEA = 0.0
'''Reference Height of sea level [m]'''

SCALE_HEIGHT = 8000.0
'''Atmospheric Scale Heigth [m]'''



########## COORDINATE FRAME/CELESTIAL BODY VARIABLES
ECI = FramesFactory.getEME2000()
'''Earth Centered Inertial Reference Frame'''

GCRF = FramesFactory.getEME2000()
'''Geocentric Celestial Reference Frame'''

ECEF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
'''Earth Centered Earth Fixed Reference Frame'''

ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
''' International Terrestrial Reference Frame'''

TOD = FramesFactory.getTOD(IERSConventions.IERS_2010, False)
'''True Equator Of Date Reference Frame'''

EARTH = OneAxisEllipsoid(EARTH_RADIUS, EARTH_FLATTENING, ECEF)
'''Earth Ellipsoid Surface Model'''

MOON = CelestialBodyFactory.getMoon()
'''Moon Celestial Body'''

SUN = CelestialBodyFactory.getSun()
'''Sun Celestial Body'''

WGS84ELLIPSOID = ReferenceEllipsoid.getWgs84(ECEF)
'''WGS 84 Ellispoid Surface Model'''

UTC = TimeScalesFactory.getUTC()
'''Coordinated Universal Time System'''



########## CONVERSION VARIABLES
MINUTE = 60
'''Number of seconds in a minute'''

HOUR = MINUTE*60
'''Number of seconds in an hour'''

DAY = HOUR*24
'''Number of seconds in a day'''

KM2M = 1000
'''Converting km to m'''

M2KM = 0.001
'''Converting m to km'''



########## GROUND STATION VARIABLES
LATITUDE = 37.1348
'''Latitude of Ground Station [Degrees]'''

LONGITUDE = -122.2110
'''Longitude of Ground Station [Degrees]'''

ALTITUDE = 684.0
'''Altitude of Ground Station [m]'''

STATION = GeodeticPoint(radians(LATITUDE),radians(LONGITUDE),ALTITUDE)
'''Observation Station Geodetic Point'''

STATION_FRAME = TopocentricFrame(EARTH, STATION, "Esrange")
'''Topocentric Reference Frame of Observatino Station'''



########## PROPAGATOR VARIABLES
MIN_STEP = 0.001
'''Minimum Propagator Step [s]'''

MAX_STEP = 1000.0
'''Maximum Propagator Step [s]'''

INIT_STEP = 100.0
'''Initial Propagator Step [s]'''

POSITION_TOLERANCE = 1.0
'''Minimum Positional Tolerance of Propagator [m]'''



########## ORBIT DETERMINATION VARIABLES
TOLERANCE = 1e-6
'''Root mean square iteration tolerance'''

MAX_ITERATIONS = 100
'''Maximum iteration count'''


########## SPACE TRACK API VARIABLES
USERNAME = 'jjohn145@calpoly.edu'
'''Username for Space Track Account'''

PASSWORD = 'E9pD9Qi!S6zqKht'
'''Username for Space Track Account'''


########## TLES

N40380D95939 = ['1 40380U 15003E   19227.49223282  .00000529  00000-0  28426-4 0  9999',
				'2 40380  99.1192 210.2314 0138825 288.8298  69.7915 15.13611928250111']

N40380D95393 = ['1 40380U 15003E   19226.43452352  .00000617  00000-0  32278-4 0  9992',
				'2 40380  99.1191 208.6663 0138853 292.3507  66.3053 15.13610872249950']

N43548D105947 = ['1 43548U 98067NW  19259.47556602  .00013669  00000-0  15221-3 0  9994',
				 '2 43548  51.6352 249.6582 0005976 106.0085 254.1568 15.63157231 67111']

N43546D111438 = ['1 43546U 98067NU  19276.92102951 +.00009177 +00000-0 +90532-4 0  9993',
				 '2 43546 051.6365 155.4308 0004242 171.0894 189.0178 15.66754740069798']

N43020D104093 = ['1 43020U 98067NH  19249.45733917  .00020505  00000-0  14835-3 0  9999',
				 '2 43020  51.6422 271.4157 0003186  64.1849 295.9478 15.73444995102400']

N43020D115159 = ['1 43020U 98067NH  19287.85731519  .00028310  00000-0  18492-3 0  9996',
				 '2 43020  51.6415  74.1103 0001638 248.6364 111.4461 15.75610335108452']

N41851D24809 = ['1 41851U 16067D   18177.21663408 -.00000234  00000-0 -15191-4 0  9995',
				'2 41851  97.9242 265.4524 0009954 128.8167 231.3944 14.96679127 88444']

N41789D96960 = ['1 41789U 16059G   19233.13987202  .00000034  00000-0  15417-4 0  9999',
				'2 41789  98.0654 293.4382 0028614 171.3394 188.8318 14.64255888154933']

N41789D99144 = ['1 41789U 16059G   19239.08502667  .00000028  00000-0  14398-4 0  9997',
				'2 41789  98.0637 299.2595 0028956 152.7270 207.5466 14.64256895155808']

N41788D93512 = ['1 41788U 16059F   19222.18682140  .00004827  00000-0  73025-3 0  9996',
				'2 41788  98.2422 298.4327 0028160 198.6286 161.3868 14.74437263153743']

N41788D92482 = ['1 41788U 16059F   19219.20080378  .00004599  00000-0  69683-3 0  9991',
				'2 41788  98.2414 295.3950 0027599 208.1768 151.7974 14.74406613153307']

N16908D106277 = ['1 16908U 86061A   19260.88154734 -.00000081 +00000-0 +11401-3 0  9993',
				 '2 16908 050.0085 164.8817 0011517 028.2674 011.4436 12.44490499173672']

N2802D62950 = ['1  2802U 67045B   18310.94306159 +.00000005 +00000-0 +96978-5 0  9993',
			   '2  2802 074.0105 301.4143 0067694 234.4983 124.9847 14.43710732702255']

N5118D62700 = ['1  5118U 71028B   18304.93143917 +.00001054 +00000-0 +54782-4 0  9996',
			   '2  5118 081.2536 298.8152 0048125 032.8792 327.5413 15.14138246579841']

N22830D61553 = ['1 22830U 93061H   18301.94183410 -.00000046 +00000-0 +15968-6 0  9993',
				'2 22830 098.8462 332.7923 0009842 270.8802 089.1252 14.31420174309346']

N5730D65014 = ['1  5730U 71119B   18319.92841590 +.00001306 +00000-0 +87219-4 0  9995',
			   '2  5730 073.8926 303.4108 0692682 082.8997 285.0327 14.02999376278146']

N3230D63153 = ['1  3230U 68040B   18311.94362594 +.00000223 +00000-0 +26726-4 0  9996',
			   '2  3230 074.0347 304.7893 0028979 122.8387 237.5591 14.90405682690067']

N29507D64807 = ['1 29507U 06046C   18318.91176518 +.00000360 +00000-0 +33077-4 0  9991',
				'2 29507 097.7714 331.7514 0048880 316.4664 043.2706 15.00330033658315']

N877D60811 = ['1   877U 64053B   18299.34271718 -.00000099  00000-0  53128-5 0  9995',
			  '2   877  65.0769 282.6087 0083509 274.5781  84.5782 14.59344536873915']

N16908D74997 = ['1 16908U 86061A   19156.87950250 -.00000116 +00000-0 -12765-3 0  9997',
				'2 16908 050.0076 124.6704 0011324 122.8852 265.9361 12.44490042160725']








