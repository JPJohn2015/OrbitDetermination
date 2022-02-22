'''
Transformations Library has functions for Reference Frame 
transformations, Coordinate system transformations and Time 
system changes.

Authors:    James Johnson

Created:    Feb-2020
'''
#Initialize OREkit and Java Virtual Machine
import orekit
vm = orekit.initVM()

#Import Orekit Libraries
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import Constants

#Import Python Libraries
from java.io import File
from datetime import datetime, timedelta
from math import *
import numpy as np
import matplotlib.pyplot as plt
from julian import from_jd

#Import Orbit Determination Libraries
from Variables import *


########## FUNCTION LIST

def localSiderealTime(datetime):
    '''Calculates Local Sidereal Time from a datetime object

    Attributes:
        datetime - datetime object
        longitude - longitude of ground station in degrees

    Returns:
        lst - local sidereal time in degrees
        gst - greenwich sidereal time in degrees
        julianDate - julian date
    '''
    #Defining Time Increments from Datetime object
    year = datetime.year
    month = datetime.month
    day = datetime.day
    hour = datetime.hour
    minute = datetime.minute
    second = datetime.second

    #Calculate Fractional Day
    ut = hour + minute/MINUTE + second/HOUR

    #Calculate Julian Date and Julian Century
    julianDate = (367*year) - floor((7*(year+floor((month+9)/12)))/4) + floor((275*month)/9) + day + 1721013.5 + (ut/24)
    julianCentury  = (julianDate-2451545.0)/36525

    #Calculate Greewich Sidereal Time in Degrees
    gst = 100.4606184 + 36000.77004*julianCentury + 0.000387933*(julianCentury**2) - 2.58*(10**-8)*(julianCentury**3)
    while gst >= 360:
        gst = gst - 360

    #Calculate Local Siderial Time in Degrees
    lst = gst + 360.98564724*(ut/24) + LONGITUDE
    while lst >= 360:
        lst = lst - 360

    return lst

def stationSite(datetime):
    '''Calculates the site vector for the ground station in ECI
    at the desired time from the latitude, longitude and altitude 

    Attributes:
        latitude - latitude of ground station in degrees
        longitude - longitude of ground station in degrees
        altitude - altitude above sea level of ground station in meters
        datetime - datetime object

    Returns:
        rSite - Vector3D of the ground station site vector
    '''
    #Define Earth Parameters as easier variables
    Re = EARTH_RADIUS
    f = EARTH_FLATTENING

    #Calculate LST
    lst = localSiderealTime(datetime)
    
    #Calculate Vector Coefficients
    C1 = ((Re/sqrt(1-(2*f-(f**2))*(sin(radians(LATITUDE))**2)))+(ALTITUDE/KM2M))*cos(radians(LATITUDE))
    C2 = (((Re*(1-f)**2)/sqrt(1-(2*f-(f**2))*(sin(radians(LATITUDE))**2)))+(ALTITUDE/KM2M))

    #Final Size Vector
    rSite = Vector3D(C1*cos(radians(lst)),C1*sin(radians(lst)),C2*sin(radians(LATITUDE)))

    return rSite

def RangeGuess(el1,el3,h):
    '''Calculates the guess for the 1st and 3rd range values
    used in Gooding's orbit determination algorithm

    Attributes:
        el1 - elevation angle at 1st measurement time in degrees
        el3 - elevation angle at 3rd measurement time in degrees
        h - guess for the orbital altitude in km

    Returns:
        rho1 - range magnitude guess at 1st measurement in meters
        rho3 - range magnitude guess at 3rd measurement in meters
    '''
    #Define Earth Parameters as easier variables
    Re = EARTH_RADIUS*M2KM

    #Estimate Slant Range Estimates for Gooding orbit determination algorithm
    rho1 = Re*(sqrt(((h+Re)/Re)**2 - cos(radians(el1))**2) - sin(radians(el1)))*(10**3)
    rho3 = Re*(sqrt(((h+Re)/Re)**2 - cos(radians(el3))**2) - sin(radians(el3)))*(10**3)

    return rho1, rho3

def PVCoordinatesToList(pv):
	'''Formats PVCoordinates object into a State Vector

	Attributes:
		pvCoordinates - PVCoordinates object

	Returns:
		X - state vector
	'''
	#Position and Velocity
	position = pv.getPosition()
	velocity = pv.getVelocity()

	#Create State Vector
	X = [position.getX(),position.getY(),position.getZ(),
		 velocity.getX(),velocity.getY(),velocity.getZ()]

	return X

def stateVectorToRaDec(X,datetime):
	'''Calculates the transformation from cartesian coordinates
	to right ascension and declination

	Attributes:
		X - state vector
        latitude - latitude of ground station in degrees
        longitude - longitude of ground station in degrees
        altitude - altitude above sea level of ground station in meters
        datetime - datetime object

	Returns:
		ra - right ascension
		dec - declination
	'''
	#Calculate Site position
	rSite = stationSite(datetime)

	#Calculate Slant Range unit vector
	slantRange = [X[0] - rSite.getX(),
				  X[1] - rSite.getY(),
				  X[2] - rSite.getZ()]
	magnitude = sqrt(slantRange[0]**2 + slantRange[1]**2 + slantRange[2]**2)
	unitVector = [slantRange[0]/magnitude,
				  slantRange[1]/magnitude,
				  slantRange[2]/magnitude]

	#Calculate Declination and Right Ascension
	dec = asin(unitVector[2])
	if unitVector[1] > 0:
		ra = acos(unitVector[0]/cos(dec))
	else:
		ra = 2*pi - acos(unitVector[0]/cos(dec))

	return degrees(ra), degrees(dec)










