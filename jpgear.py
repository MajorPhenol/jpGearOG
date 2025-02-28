"""Gear Designer

This module allows the user to design a pair of spur gears in mesh.  The designs
can then be exported as .DXF drawings.

First, create a pair of 'Gear' objects, then pass them into a 'Mesh' object.
Once the mesh is created, the individual gear parameters can be tweaked.

Example:
> import jpgear
> from jpgear import Gear as Gear
> from jpgear import Mesh as Mesh
> g1 = Gear(20, 1.0)
> g2 = Gear(42, 1.0)
> m = Mesh(g1, g2)
> m.setBacklash(0.3)
> m.setRootClearances([0.2, 0.2])
> m.drawMesh(animate=True)
> m.drawGear(1, save="output.dxf")

Required packages:
	numpy
	scipy
	matplotlib
	ezdxf

TODO
- double check stress
"""

from numpy import pi, sin, cos, tan, arcsin, arccos, arctan
from numpy import rad2deg, deg2rad
from numpy import sqrt, zeros, linspace, polyval, real

from scipy.optimize import fsolve

import matplotlib.pyplot as pyplot
import matplotlib.path as mpath
import matplotlib.patches as mpatch
import matplotlib.collections as mcollections
import matplotlib.transforms as mtransforms
import matplotlib.animation as manimation
from matplotlib.widgets import Slider

import copy

import ezdxf
# from ezdxf.gfxattribs import GfxAttribs
from ezdxf.math import UCS
from ezdxf import zoom

###############################################################################
# why isn't this in numpy???
tau = 2*pi

###############################################################################

# Involute Function
def invF(angle):
	"""Computes the involute function of a given profile angle.

	All angles in radians.
	"""

	return tan(angle) - angle

# Reverse Involute Function
def revInvF(angle):
	"""Computes the reverse involute function, returning the profile angle.
	
	From "The Geometry of Involute Gears" by J.R. Colbourne
	Eqs. 2.16 - 2.17
	
	All angles in radians.
	"""

	q = angle**(2/3)

	poly = [-0.00048, 0.00319, -0.00894, -0.00321, 0.32451, 1.04004, 1]
	x = polyval(poly, q)

	return arccos(1/x)

###############################################################################

class Gear():
	"""The gear object, containing all the gear dimensions.

	Args
	----
	N (int) : number of teeth
	mod (float) : module, mm
	PA_deg (float) : pressure angle, degrees (default=20)

	Note
	----
	Currently, only metric modules are supported.
	"""

	def __init__(self, _N, _mod, _PA_deg=20):
		self.N = _N
		self.mod = _mod
		self.PA_deg = _PA_deg              		# pressure angle, degrees
		self.PA = deg2rad(self.PA_deg)      	# pressure angle, radians
		
		self.Rs = 0.5 * self.N * self.mod		# standard pitch radius
		self.Rp = self.Rs						# pitch radius at OPA
		self.Rb = self.Rs * cos(self.PA)		# base circle radius
		self.Pb = 2*pi * self.Rs / self.N		# base pitch

		self.Ros = (self.mod * (self.N+2) / 2)	# standard outer radius
		self.Ro = self.Ros						# actual outer radius
		self.Rtip = 0							# tip radius
		self.Rtip_max = 0 						# max possible tip radius
		self.Roe = self.Ro						# effective outer radius, considering tip radius
		self.updateRoe()

		self.Rr = self.Rb						# root radius
		self.Rf = 0								# root fillet radius
		self.Rff = 0							# root full fillet radius
		self.theta_F = 0						# angle between tooth centerline and root fillet center
		self.phi_JFI = 0						# profile angle at the junction of root fillet and involute, radians

		self.x = 0								# profile shift
		self.tts = self.mod * (pi/2)			# standard tooth thickness
												# Note - this will get updated for any profile shift
		self.tt = self.tts						# tooth thickness at Rp
		self.Rhp = 0							# highest point of single tooth contact

		self.FW = 1								# face width

		self.undercut = False

	def updateRoe(self):
		self.Roe = sqrt( self.Rb**2 +
				( sqrt((self.Ro-self.Rtip)**2 - self.Rb**2) + self.Rtip )**2 )

###############################################################################

class Mesh():
	"""The mesh object, containing the dimensions of the gear mesh.

	Args
	----
	gear1, gear2 : Gear objects

	Note
	----
	'gear1' is the driving gear, 'gear2' is the driven gear.

	User Methods
	------------
	setBacklash()
	setProfileShifts()
	setRootClearances()
	setRootFillets()
	setTipRadius()
	drawGear()
	drawMesh()
	stress()
	"""

	def __init__(self, _gear1, _gear2):
		if _gear1.mod != _gear2.mod:
			raise ValueError("Gears must have the same module!\n")

		self.G1 = _gear1		# gear objects
		self.G2 = _gear2

		self.bkl = 0 			# backlash
		self.rtcl1 = 0 			# root clearances
		self.rtcl2 = 0

		self.setBacklash(self.bkl)
		self.setRootClearances([self.rtcl1, self.rtcl2])

		self.findMaxTipRadius(self.G1)
		self.findMaxTipRadius(self.G2)

	def setBacklash(self, _bkl):
		"""Set desired backlash.
		
		Args
		----
		bkl (float) : desired backlash, mm
		"""

		self.bkl = _bkl
		self.updateCenterDistance()

	def setProfileShifts(self, _x):
		"""Set desired profile shift for each gear.

		Args
		----
		x ([float, float]): a list containing the desired profile shifts for each gear

		Note
		----
		Passing an empty list will leave a gear unchanged, e.g.:
		> setProfileShifts([0.3, []])
		will update the first gear, and leave the second gear unchanged.
		"""

		for x, gear in zip(_x, [self.G1, self.G2]):
			if x != []:
				gear.x = x
				gear.tts = gear.mod * (pi/2 + 2*gear.x*tan(gear.PA))
		
		self.updateCenterDistance()
		self.setRootClearances([self.rtcl1, self.rtcl2])
		self.setRootFillets([self.G1.Rf, self.G2.Rf])

	def setRootClearances(self, _rtcl):
		"""Set desired root clearance for each gear.

		Args
		----
		rtcl ([float, float]): a list containing the desired root clearances for each gear

		Note
		----
		Passing an empty list will leave a gear unchanged, e.g.:
		> setRootClearances([0.3, []])
		will update the first gear, and leave the second gear unchanged.
		"""

		if _rtcl[0] != []:
			self.rtcl1 = _rtcl[0]
			self.G1.Rr = self.CD - self.G2.Ro - self.rtcl1

			self.findRootFullFillet(self.G1)
			if self.G1.Rf > self.G1.Rff:
				print("Warning: root fillet is too large, setting full fillet.")
				self.G1.Rf = self.G1.Rff
			self.checkUndercut(self.G1)
			self.updateJFI(self.G1)

		if _rtcl[1] != []:
			self.rtcl2 = _rtcl[1]
			self.G2.Rr = self.CD - self.G1.Ro - self.rtcl2
			
			self.findRootFullFillet(self.G2)
			if self.G2.Rf > self.G2.Rff:
				print("Warning: root fillet is too large, setting full fillet.")
				self.G2.Rf = self.G2.Rff
			self.checkUndercut(self.G2)
			self.updateJFI(self.G2)

	def setRootFillets(self, _Rf):
		"""Set desired root fillets for each gear.

		Args
		----
		Rf ([float|str*, float|str*]): a list containing the desired root fillets for each gear
		* Passing 'full' will create a full fillet between each pair of teeth

		Note
		----
		Passing an empty list will leave a gear unchanged, e.g.:
		> setRootFillets([0.3, []])
		will update the first gear, and leave the second gear unchanged.
		"""

		for Rf, gear in zip(_Rf, [self.G1, self.G2]):
			if Rf != []:
				if Rf == 'full':
					gear.Rf = gear.Rff
				elif Rf > gear.Rff:
					print("Warning: root fillet is too large, setting full fillet.")
					gear.Rf = gear.Rff
				else:
					gear.Rf = Rf
				self.checkUndercut(gear)
				# self.findRootFullFillet(gear)
				self.updateJFI(gear)

	def setOuterRadius(self, _Ro):
		"""Set desired outer radius for each gear.

		Args
		----
		Ro ([float, float]): a list containing the desired outer radius for each gear

		Note
		----
		Passing an empty list will leave a gear unchanged, e.g.:
		> setOuterRadius([30, []])
		will update the first gear, and leave the second gear unchanged.
		"""

		for Ro, gear in zip(_Ro, [self.G1, self.G2]):
			if Ro != []:
				# max OD is when theta_A is 0, i.e. the involute hits the tooth centerline
				# theta_A = (gear.tts / (2*gear.Rs)) + invF(gear.PA) - invF(phi_A)
				# 0 = (gear.tts / (2*gear.Rs)) + invF(gear.PA) - invF(phi_A)
				# invF(phi_A) = (gear.tts / (2*gear.Rs)) + invF(gear.PA)
				phi_A = revInvF((gear.tts / (2*gear.Rs)) + invF(gear.PA))
				Romax = gear.Rb / cos(phi_A)
				if Ro > Romax:
					print("Warning - outer radius too large, using max radius")
					gear.Ro = Romax
				else:
					gear.Ro = Ro
				gear.updateRoe()

		self.setRootClearances([self.rtcl1, self.rtcl2])

	def setTipRadius(self, _Rtip):
		"""Set desired tooth tip radius for each gear.

		Args
		----
		Rtip ([float, float]): a list containing the desired tip radius for each gear

		Note
		----
		Passing an empty list will leave a gear unchanged, e.g.:
		> setTipRadius([0.3, []])
		will update the first gear, and leave the second gear unchanged.
		"""

		for Rtip, gear in zip(_Rtip, [self.G1, self.G2]):
			if Rtip != []:
				self.findMaxTipRadius(gear)
				if Rtip > gear.Rtip_max:
					print("Warning - tip radius too large, setting to max tip radius")
					gear.Rtip = gear.Rtip_max
				else:
					gear.Rtip = Rtip
				gear.updateRoe()

	def findMaxTipRadius(self, _gear):
		"""
		##########################################
		A : point on involute where fillet starts
		E : tangent point on base circle, determined by A and phi_A
		C : center point of gear
		F : center point of fillet
		
		##########################################
		# angle between A and tooth centerline
		theta_A = (tts/(2*Rs)) + invF(PA) - invF(phi_A)
		
		# angle between line CE and tooth centerline
		alpha = phi_A - theta_A

		# length of line from E to A
		EA = Rb*tan(phi_A)

		# length of line from E to F
		EF = Rb*tan(alpha)

		# the tip fillet radius is the difference between EA and EF
		RTip1 = EA - EF
		Rtip = Rb*tan(phi_A) - Rb*tan(alpha)
		Rtip = Rb*tan(phi_A) - Rb*tan(phi_A - theta_A)
		Rtip = Rb*tan(phi_A) - Rb*tan(phi_A - (tts/(2*Rs)) - invF(PA) + invF(phi_A))

		# length of line from C to F
		CF = Rb / cos(alpha)

		# the tip fillet radius is the difference between Ro and CF
		Rtip2 = Ro - CF
		Rtip2 = Ro - (Rb / cos(alpha))
		Rtip2 = Ro - (Rb / cos(phi_A - theta_A))
		Rtip2 = Ro - (Rb / cos(phi_A - (tts/(2*Rs)) - invF(PA) + invF(phi_A)))

		"""

		Ro = _gear.Ro
		Rb = _gear.Rb
		Rs = _gear.Rs
		tts = _gear.tts
		PA = _gear.PA

		def RTip1(phi_A):
			return Rb*tan(phi_A) - Rb*tan(phi_A - (tts/(2*Rs)) - invF(PA) + invF(phi_A))

		def RTip2(phi_A):
			return Ro - (Rb / (cos(phi_A - (tts/(2*Rs)) - invF(PA) + invF(phi_A))))

		# RTip1 and RTip2 are equal, so this should be zero
		def func(phi_A):
			return RTip1(phi_A) - RTip2(phi_A)

		# initial guess is at standard pitch radius
		initialGuess = arccos(Rb/Rs)
		phi_A_solved = fsolve(func, initialGuess).item()
		_gear.Rtip_max = RTip1(phi_A_solved)

	def updateCenterDistance(self):
		[R1, R2, tt1, tt2] = self.findCenterDistance()
		self.CD = R1.item() + R2.item()
		self.G1.tt = tt1.item()
		self.G2.tt = tt2.item()
		# Operating pressure angle at updated center distance
		self.OPA = arccos((self.G1.Rb+self.G2.Rb) / self.CD)
		self.OPA_deg = rad2deg(self.OPA)
		# update pitch radius
		self.G1.Rp = self.G1.Rb/cos(self.OPA)
		self.G2.Rp = self.G2.Rb/cos(self.OPA)
		
		self.updateContactRatio()

	def updateContactRatio(self):
		# AGMA-908 Parameters
		self.C6 = self.CD * sin(self.OPA)
		self.C1 = self.C6 - sqrt(self.G2.Roe**2 - self.G2.Rb**2)
		self.C5 = sqrt(self.G1.Roe**2 - self.G1.Rb**2)
		self.C2 = self.C5 - self.G1.Pb
		self.C3 = (self.G1.N/(self.G1.N+self.G2.N)) * self.C6
		self.C4 = self.C1 + self.G1.Pb

		# Line of contact
		self.LoC = self.C5 - self.C1;

		# Contact ratio
		self.CR = self.LoC / self.G1.Pb;

		# Highest point of single tooth contact
		self.G1.Rhp = sqrt(self.G1.Rb**2 + self.C4**2);
		self.G2.Rhp = sqrt(self.G2.Rb**2 + (self.C6-self.C2)**2);

	def findCenterDistance(self):
		N1 = self.G1.N
		N2 = self.G2.N

		#   Precalculate involute function at standard pitch / PA
		invS = invF(self.G1.PA)

		#   Standard pitch radius (no shift)
		Rs1 = self.G1.Rs
		Rs2 = self.G2.Rs

		#   Base circle radius
		Rb1 = self.G1.Rb
		Rb2 = self.G2.Rb

		#   Standard tooth thickness with profile shift
		mod = self.G1.mod
		tts1 = self.G1.tts
		tts2 = self.G2.tts
		bkl = self.bkl

		def func(x):
			# variables for function: R1, R2, tt1, tt2
			F = zeros(4)

			# pitch circle is determined by mod and number of teeth => mod = R/N
			# both pinion and gear must have same mod, so R1/N1 = R2/N2
			# => N2*R1 - N1*R2 = 0			
			F[0] = N2*x[0] - N1*x[1]
			# circular pitch is sum of each tooth thickness and backlash
			# (2*pi*R1)/N1 = tt1 + tt2 + bkl [equivalently, (2*pi*R2)/N2 = tt1 + tt2 + bkl]
			# => (2*pi*R1)/N1 - tt1 - tt2 - bkl = 0
			F[1] = (2*pi*x[0])/N1 - x[2] - x[3] - bkl
			# calculate tooth thickness at new pitch radius
			# tt = R*( (tts/Rs) + 2*(invF(PA) - invF(acos(Rb/R)) ) )
			# => R*( (tts/Rs) + 2*(invF(PA) - invF(acos(Rb/R)) ) ) - tt = 0
			F[2] = x[0]*( (tts1/Rs1) + 2*(invS - invF(arccos(Rb1/x[0])) ) ) - x[2]
			F[3] = x[1]*( (tts2/Rs2) + 2*(invS - invF(arccos(Rb2/x[1])) ) ) - x[3]

			return F

		#   For initial guess, use standard values
		initialGuess = [Rs1, Rs2, tts1, tts2]
		root = fsolve(func, initialGuess)

		return root

	def checkUndercut(self, _gear):
		# distance from the gear center to the center point of the root fillet,
		# assuming the JFI is on the base circle
		# this forms a right triangle with Rb and Rf
		CF = sqrt(_gear.Rb**2 + _gear.Rf**2)
		# CF sets the minimum Rr for a given Rf
		Rrmin = CF - _gear.Rf

		# Undercut check
		if _gear.Rr < Rrmin:
			_gear.undercut = True
			print('Warning -', _gear.N, 'tooth gear is undercut')
		else:
			_gear.undercut = False

	def findRootFullFillet(self, _gear):
		###########################################################################
		#   Calculate fillet radius. This function computes two separate distances:
		#   1.) the distance from the fillet center to the involute curve, and
		#   2.) the distance from the fillet center to the root circle
		#   The function then finds the condition where these two distances are the
		#   same.

		# _gear - gear object
		N = _gear.N			#number of teeth
		tts = _gear.tts		# tooth thickness at standard pitch radius, mm
		Rs = _gear.Rs		# standard pitch radius, mm
		Rb = _gear.Rb		# base radius, mm
		Rr = _gear.Rr		# root radius, mm
		PA = _gear.PA		# standard pressure angle, radians

		# phi_A - profile angle at some point A on the involute
		# theta_A - angle between tooth centerline and some point A on the involute
		# phi_F - angle between fillet centerline and Rb, where a line tangent to
		#   the base circle goes through some point A on the involute
		# Rf_1 - distance between fillet centerline and some point A on the
		#   involute, along a line tangent to the base circle
		# Rf_2 - distance between the fillet center and the root circle
		# JFI - junction of fillet and involute
		###########################################################################

		# angle between tooth centerline and involute at base circle (phi_A = 0)
		theta_A = tts/(2*Rs) + invF(PA)
		# angle between involute at base circle and center of tooth gap
		alpha = pi/N - theta_A
		# full fillet radius assuming the JFI is on the base circle
		Rfu = Rb * tan(alpha)
		Rrmin = Rb - Rfu
		
		# Undercut
		if Rr < Rrmin:
			_gear.Rff = -(Rr*sin(alpha))/(sin(alpha) - 1)
		else:
			# angle between tooth centerline and fillet centerline = pi/N
			# phi_F = pi/N - theta_A + phi_A
			# where: theta_A = tts/(2*Rs) + invF(PA) - invF(phi_A)
			# where: invF(phi_A) = tan(phi_A) - phi_A
			# => phi_F = pi/N - (tts/(2*Rs) + invF(PA) - (tan(phi_A) - phi_A)) + phi_A
			def phi_F(phi_A):
				return pi/N - tts/(2*Rs) - invF(PA) + tan(phi_A)

			def Rf1(phi_A):
				return Rb*(tan(phi_F(phi_A)) - tan(phi_A))
			
			def Rf2(phi_A):
				return Rb/cos(phi_F(phi_A)) - Rr

			# Rf1 and Rf2 are equal, so this should be zero
			def func(phi_A):
				return Rf1(phi_A) - Rf2(phi_A)

			# initial guess is at standard pitch radius
			initialGuess = arccos(Rb/Rs)
			phi_JFI = fsolve(func, initialGuess).item()

			# _gear.phi_JFI = phi_JFI
			_gear.Rff = Rf1(phi_JFI)

	def updateJFI(self, _gear):

		if _gear.undercut == True:
			_gear.phi_JFI = 0

			theta_A = (_gear.tts/(2*_gear.Rs)) + invF(_gear.PA)
			# angle between JFI and center of fillet circle
			alpha_F = arcsin(_gear.Rf / (_gear.Rr + _gear.Rf))
			_gear.theta_F = theta_A + alpha_F
		else:
			# profile angle through center of fillet circle
			phi_F = arccos(_gear.Rb / (_gear.Rr + _gear.Rf))
			# line tangent to base circle through fillet center point
			EF = sqrt((_gear.Rr+_gear.Rf)**2 - _gear.Rb**2)
			# line tangent to base circle to involute
			EA = EF - _gear.Rf
			# profile angle at JFI
			phi_A = arctan(EA / _gear.Rb)
			theta_A = (_gear.tts/(2*_gear.Rs)) + invF(_gear.PA) - invF(phi_A)
			theta_F = phi_F - phi_A + theta_A

			_gear.phi_JFI = phi_A
			_gear.theta_F = theta_F

	def layoutGear(self, _gear, _save=None):
		G = _gear
		savePath = _save

		circleList = []

		# base circle
		circleList.append(pyplot.Circle((0, 0), G.Rb, color='c', ls='--', fill=False))
		# pitch circle
		circleList.append(pyplot.Circle((0, 0), G.Rp, color='y', ls='--', fill=False))
		# outer circle
		circleList.append(pyplot.Circle((0, 0), G.Roe, color='g', ls='--', fill=False))
		# root cirlce
		circleList.append(pyplot.Circle((0, 0), G.Rr, color='r', ls='--', fill=False))

		curveList = []
		curveColor = 'b'
		curveWidth = 2
		
		# Involute
		# find staring point of involute
		Rjfi = G.Rb/(cos(G.phi_JFI));
		theta_JFI = G.tts/(2*G.Rs) + invF(G.PA) - invF(G.phi_JFI);
		Rjfi_x = Rjfi*sin(theta_JFI);
		Rjfi_y = Rjfi*cos(theta_JFI);
		# create vectors of points along the involute
		RA = linspace(Rjfi, G.Roe, 10)
		phi_A = arccos(G.Rb/RA)
		theta_A = (G.tts/(2*G.Rs)) + invF(G.PA) - invF(phi_A)
		RAx = RA*sin(theta_A)
		RAy = RA*cos(theta_A)
		# add right side
		invPath = mpath.Path( list(map(list, zip(*[RAx, RAy]))) )
		invPatch = mpatch.PathPatch(invPath, color=curveColor, linewidth=curveWidth, fill=False)
		curveList.append(invPatch)
		# add left side
		invPath = mpath.Path( list(map(list, zip(*[-RAx, RAy]))) )
		invPatch = mpatch.PathPatch(invPath, color=curveColor, linewidth=curveWidth, fill=False)
		curveList.append(invPatch)

		# Root fillet radius
		if G.Rf > 0:
			# find center of fillet circle
			Fx = (G.Rr + G.Rf)*sin(G.theta_F)
			Fy = (G.Rr + G.Rf)*cos(G.theta_F)
			# find arc angles
			filletStartAngle = 180 + rad2deg(G.phi_JFI - theta_JFI);
			filletEndAngle = 360 - 90 - rad2deg(G.theta_F);
			curveList.append(mpatch.Arc(
										(Fx,Fy),
										G.Rf*2, G.Rf*2,
										theta1=filletStartAngle, theta2=filletEndAngle,
										color=curveColor,
										linewidth=curveWidth
										)
							)
			curveList.append(mpatch.Arc(
										(-Fx,Fy),
										G.Rf*2, G.Rf*2,
										theta1=180-filletEndAngle, theta2=180-filletStartAngle,
										color=curveColor,
										linewidth=curveWidth
										)
							)

		# add straight line segment if undercut
		if G.undercut == True:
			theta_A = G.tts/(2*G.Rs) + invF(G.PA)
			Rjfi = (G.Rr + G.Rf) * cos(G.theta_F - theta_A)
			Rjfi_x = Rjfi*sin(theta_A)
			Rjfi_y = Rjfi*cos(theta_A)

			Rb_x = G.Rb*sin(theta_A)
			Rb_y = G.Rb*cos(theta_A)

			line = mpath.Path([[Rjfi_x, Rjfi_y], [Rb_x, Rb_y]])
			linePatch = mpatch.PathPatch(line, color=curveColor, linewidth=curveWidth, fill=False)
			curveList.append(linePatch)
			line = mpath.Path([[-Rjfi_x, Rjfi_y], [-Rb_x, Rb_y]])
			linePatch = mpatch.PathPatch(line, color=curveColor, linewidth=curveWidth, fill=False)
			curveList.append(linePatch)

		# Root Radius
		if G.Rf < G.Rff:
			RRStartAngle1 = rad2deg(G.theta_F) + 90
			RREndAngle1 = rad2deg(pi/G.N) + 90
			RRStartAngle2 = -rad2deg(pi/G.N) + 90
			RREndAngle2 = -rad2deg(G.theta_F) + 90
			curveList.append(mpatch.Arc(
							(0, 0),
							G.Rr*2, G.Rr*2,
							theta1=RRStartAngle1, theta2=RREndAngle1,
							color=curveColor,
							linewidth=curveWidth
							)
				)
			curveList.append(mpatch.Arc(
							(0, 0),
							G.Rr*2, G.Rr*2,
							theta1=RRStartAngle2, theta2=RREndAngle2,
							color=curveColor,
							linewidth=curveWidth
							)
				)

		# Tip Radius
		if G.Rtip > 0:
			# point where tip touches involute
			phi_Atip = arccos(G.Rb/G.Roe);
			theta_Atip = (G.tts/(2*G.Rs)) + invF(G.PA) - invF(phi_Atip);
			# point where tip touches OD
			CF = G.Ro - G.Rtip; # distance from gear center to fillet center
			phi_Otip = arccos(G.Rb/CF);
			theta_Otip = theta_Atip - phi_Atip + phi_Otip;
			# Center point
			Rtip_x = CF*sin(theta_Otip);
			Rtip_y = CF*cos(theta_Otip);
			# find arc angles
			tipStartAngle = rad2deg(phi_Atip - theta_Atip);
			tipEndAngle = 90 - rad2deg(theta_Otip);
			curveList.append(mpatch.Arc(
										(Rtip_x,Rtip_y),
										G.Rtip*2, G.Rtip*2,
										theta1=tipStartAngle, theta2=tipEndAngle,
										color=curveColor,
										linewidth=curveWidth
										)
							)
			curveList.append(mpatch.Arc(
										(-Rtip_x,Rtip_y),
										G.Rtip*2, G.Rtip*2,
										theta1=180-tipEndAngle, theta2=180-tipStartAngle,
										color=curveColor,
										linewidth=curveWidth
										)
							)

		# Outer Radius
		if G.Rtip < G.Rtip_max:
			if 'theta_Otip' in locals():
				theta_O = theta_Otip
			else:
				phi_O = arccos(G.Rb/G.Roe);
				theta_O = (G.tts/(2*G.Rs)) + invF(G.PA) - invF(phi_O);
			ODStartAngle = -rad2deg(theta_O) + 90
			ODEndAngle = rad2deg(theta_O) + 90
			curveList.append(mpatch.Arc(
										(0,0),
										G.Ro*2, G.Ro*2,
										#angle=90,
										theta1=ODStartAngle, theta2=ODEndAngle,
										color=curveColor,
										linewidth=curveWidth
										)
							)

		# make copies of all the teeth
		fullList = []
		for n in range(G.N):
			angle = (n/G.N)*tau
			for curve in curveList:
				newCurve = copy.copy(curve)
				newCurve.set_transform(mtransforms.Affine2D().rotate(angle))
				fullList.append(newCurve)

		# circles
		circleCollection = mcollections.PatchCollection(circleList, match_original=True)
		# teeth
		curveCollection = mcollections.PatchCollection(fullList, match_original=True)

		if savePath:

			doc = ezdxf.new()
			msp = doc.modelspace()

			# base circle
			# msp.add_circle((0,0), radius = G.Rb, dxfattribs=GfxAttribs(color=ezdxf.colors.CYAN))
			# pitch circle
			# msp.add_circle((0,0), radius = G.Rp, dxfattribs=GfxAttribs(color=ezdxf.colors.YELLOW))
			# outer circle
			# msp.add_circle((0,0), radius = G.Roe, dxfattribs=GfxAttribs(color=ezdxf.colors.GREEN))

			for n in range(G.N):
				angle = (n/G.N)*tau
				ucs = UCS(origin=(0,0,0)).rotate_local_z(angle)

				# involute
				msp.add_lwpolyline(list(zip(RAx, RAy))).transform(ucs.matrix)
				msp.add_lwpolyline(list(zip(-RAx, RAy))).transform(ucs.matrix)

				# root fillet
				if G.Rf > 0:
					msp.add_arc((Fx, Fy), radius=G.Rf, start_angle=filletStartAngle, end_angle=filletEndAngle).transform(ucs.matrix)
					msp.add_arc((-Fx, Fy), radius=G.Rf, start_angle=180-filletEndAngle, end_angle=180-filletStartAngle).transform(ucs.matrix)

				# add straight line segment if undercut
				if G.undercut == True:
					msp.add_line((Rjfi_x, Rjfi_y), (Rb_x, Rb_y)).transform(ucs.matrix)
					msp.add_line((-Rjfi_x, Rjfi_y), (-Rb_x, Rb_y)).transform(ucs.matrix)

				# root radius
				if G.Rf < G.Rff :
					msp.add_arc((0, 0), radius=G.Rr, start_angle=RRStartAngle1, end_angle=RREndAngle1).transform(ucs.matrix)
					msp.add_arc((0, 0), radius=G.Rr, start_angle=RRStartAngle2, end_angle=RREndAngle2).transform(ucs.matrix)

				# tip radius
				if G.Rtip > 0:
					msp.add_arc((Rtip_x,Rtip_y), radius=G.Rtip, start_angle=tipStartAngle, end_angle=tipEndAngle).transform(ucs.matrix)
					msp.add_arc((-Rtip_x,Rtip_y), radius=G.Rtip, start_angle=180-tipEndAngle, end_angle=180-tipStartAngle).transform(ucs.matrix)

				# outer radius
				if G.Rtip < G.Rtip_max:
					msp.add_arc((0, 0), radius=G.Ro, start_angle=ODStartAngle, end_angle=ODEndAngle).transform(ucs.matrix)

			zoom.extents(msp)
			doc.saveas(str(savePath))

		return curveCollection, circleCollection

	def drawGear(self, _gearNumber, circles=False, save=""):
		"""Draws a single gear.

		Args
		----
		gearNumber (1 or 2) : which gear to draw
		circles (bool) : display the base (cyan), pitch (yellow),
						 outer (green), and root (red) circles (default=False)
		save (str) : filename to save a DXF (optional, default=empty)
		"""

		if _gearNumber == 1:
			curveCol, circleCol = self.layoutGear(self.G1, save)
			G = self.G1
		elif _gearNumber == 2:
			curveCol, circleCol = self.layoutGear(self.G2, save)
			G = self.G2
		else:
			print("ERROR: Enter 1 or 2\n")
			return

		figure, axes = pyplot.subplots()
		limit = G.Ro * 1.1
		axes.set_xlim(left=-limit, right=limit)
		axes.set_ylim(bottom=-limit, top=limit)
		axes.set_aspect('equal')

		axes.add_collection(curveCol)
		if circles:
			axes.add_collection(circleCol)

		pyplot.show()

	def drawMesh(self, circles=False, trace=False, animate=False, speed=10):
		"""Draws both gears in mesh.

		Args
		----
		circles (bool) : display the base (cyan), pitch (yellow),
						 outer (green), and root (red) circles (default=False)
		trace (bool) : display the path of the point of contact (default=False)
		animate (bool) : make 'em spin (default=False)
		speed (float) : animation speed (default=10)
		"""

		px = 1/pyplot.rcParams['figure.dpi']
		figure, axes = pyplot.subplots(figsize=(1200*px, 900*px))
		buffer = 1.1
		limitLeft = self.G1.Ro * buffer
		limitRight = self.CD + self.G2.Ro * buffer
		limitV = max(self.G1.Ro, self.G2.Ro) * buffer
		axes.set_xlim(left=-limitLeft, right=limitRight)
		axes.set_ylim(bottom=-limitV, top=limitV)
		axes.set_aspect('equal')

		curveCol1, circleCol1 = self.layoutGear(self.G1)
		curveCol2, circleCol2 = self.layoutGear(self.G2)
		curveCol2.set_edgecolor('r')

		# make the teeth mesh nicely
		ratio = self.G1.N / self.G2.N
		# profile angle at pitch circle
		phi_P1 = arccos(self.G1.Rb/self.G1.Rp)
		phi_P2 = arccos(self.G2.Rb/self.G2.Rp)
		# tooth thickness at pitch circle
		theta_P1 = self.G1.tt/(2*self.G1.Rp)
		theta_P2 = self.G2.tt/(2*self.G2.Rp)

		startAngle1 = -pi/2 + theta_P1
		startAngle2 = pi/2 + theta_P2

		curveCol1.set_transform(mtransforms.Affine2D().rotate(startAngle1) + axes.transData)
		curveCol2.set_transform(mtransforms.Affine2D().rotate(startAngle2).translate(self.CD, 0) + axes.transData)

		axes.add_collection(curveCol1)
		axes.add_collection(curveCol2)
		
		if circles:
			axes.add_collection(circleCol1)
			circleCol2.set_transform(mtransforms.Affine2D().translate(self.CD, 0) + axes.transData)
			axes.add_collection(circleCol2)

		# Points for line of contact
		# pitch point
		x0 = self.G1.Rp
		y0 = 0
		# tangent to gear 1
		x1 = self.G1.Rb*cos(self.OPA)
		y1 = self.G1.Rb*sin(self.OPA)
		# tangent to gear 2		
		x2 = self.CD - self.G2.Rb*cos(self.OPA)
		y2 = -self.G2.Rb*sin(self.OPA)
		# on LoC at Roe2
		# law of sines - A/sin(a) = B/sin(b)
		b = arcsin( (self.G2.Rp/self.G2.Roe) * sin(self.OPA + deg2rad(90)) )
		c = pi - b - self.OPA - deg2rad(90)
		x3 = self.CD - (self.G2.Roe * cos(c))
		y3 = self.G2.Roe * sin(c)
		b = arcsin( (self.G1.Rp/self.G1.Roe) * sin(self.OPA + deg2rad(90)) )
		c = pi - b - self.OPA - deg2rad(90)
		x4 = (self.G1.Roe * cos(c))
		y4 = -(self.G1.Roe * sin(c))

		if not animate:
			sliderAxis = figure.add_axes([0.15, 0.05, 0.70, 0.03])
			
			sliderMin = 0
			# pitch point happens at slider=1
			# find overshoot for gear two
			a = arctan(-y2/x2)
			overshoot = a / self.OPA
			sliderMax = 1 + overshoot
			slider = Slider(ax=sliderAxis, label='', valmin=sliderMin, valmax=sliderMax, valinit=1)

			if trace:
				# line of contact
				axes.plot([x1,x2],[y1,y2], color='k', marker='x', linestyle='--', linewidth=1)
				axes.plot([x3,x4],[y3,y4], color='k', marker='x', linewidth=2)
				# contact point
				tracePoint, = axes.plot([self.G1.Rp], [0], color='tab:orange', marker='o', linewidth=2)

			def update(val):
				phi_A = val * self.OPA
				updateAngle = phi_A + invF(phi_A)

				curveStartAngle1 = startAngle1 + self.OPA + invF(self.OPA)
				curveStartAngle2 = startAngle2 - ratio * (self.OPA + invF(self.OPA))

				angle1 = curveStartAngle1 - (updateAngle)
				angle2 = curveStartAngle2 + (updateAngle * ratio)
				
				if trace:
					traceAngle = self.OPA - phi_A
					R = self.G1.Rb/cos(phi_A)
					x = R*cos(traceAngle)
					y = R*sin(traceAngle)
					tracePoint.set_data([x], [y])

				curveCol1.set_transform(mtransforms.Affine2D().rotate(angle1)+ axes.transData)
				curveCol2.set_transform(mtransforms.Affine2D().rotate(angle2).translate(self.CD, 0) + axes.transData)

			slider.on_changed(update)

		if animate:
			# 'speed' is passed in, RPM
			msPerRev = (60*1000)/speed 			# milliseconds per revolution
			interval = 33						# ms per frame
			framesPerRev = int(msPerRev / interval)	# frames for one rev

			if trace:
				# line of contact
				axes.plot([x1,x2],[y1,y2], color='k', marker='x', linestyle='--', linewidth=1)
				axes.plot([x3,x4],[y3,y4], color='k', marker='x', linewidth=2)
			
			def animFunc(frame):
				angle1 = (-pi/2) - (tau*frame)/framesPerRev
				angle2 = pi/2 + pi/self.G2.N - 0.5*self.bkl/self.G2.Rs + (tau*ratio*frame)/framesPerRev
				curveCol1.set_transform(
								mtransforms.Affine2D().rotate(angle1)
								+ axes.transData
							)
				curveCol2.set_transform(
								mtransforms.Affine2D().rotate((angle2)).translate(self.CD, 0)
								+ axes.transData
							)
			anim = manimation.FuncAnimation(figure, animFunc, frames=framesPerRev, interval=interval)

		pyplot.show()

###############################################################################

	def lewisParabola(self, _gear, _Rd):

		# Find the intersection point of the Lewis parabola and the root fillet circle
		#
		#   Parameters:
		# Fx, Fy - centerpoint of root fillet circle
		# Rf - root fillet radius
		# Rd - intersection of tooth centerline and force vector; the top of the Lewis
		#   parabola
		#
		#   Output variables:
		# x, y - point of intersection between the Lewis parabola and the root fillet
		# a - scale of parabola, => y = ax^2 + bx + c
		#-------------------------------------------------------------------------------

		# Equation of a circle: (y-v)^2 + (x-h)^2 = r^2
		# Substitute values for fillet circle: (y-Fy)^2 + (x-Fx)^2 = Rf^2
		# Rearrange: y = +/-sqrt(Rf^2 - (x-Fx)^2) + Fy
		# We are only interested in the bottom half of the circle:
		# [1] y = -sqrt(Rf^2 - (x-Fx)^2) + Fy
		# Implicit differentiation of [1]:
		# [2] dy/dx = -(x-Fx)/(y-Fy)
		#
		# Equation of a parabola: y = ax^2 + bx + c
		# Substitute values of Lewis parabola, and note that the parabola is
		# centered on the y-axis:
		# [3] y = ax^2 + Rd
		# Differentiate [3]:
		# [4] dy/dx = 2ax
		#
		# At the tangent intersection of the parabola and circle, the derivatives
		# are equal:
		# [5] -(x-Fx)/(y-Fy) = 2ax
		# Rearrange [5]:
		# [6] a = -(x-Fx)/(2x)(y-Fy)
		# Substitute [6] into [3]:
		# y = -(x)(x-Fx)/(2)(y-Fy) + Rd
		# y^2 - Fy*y - Rd*y = (-x^2 + Fx*x)/2 - Rd*Fy
		# Complete the square:
		# [y^2 -(Fy+Rd)*y + (Fy+Rd)^2/4] - (Fy+Rd)^2/4 = (-x^2 + Fx*x)/2 - Rd*Fy
		# [y - (Fy+Rd)/2]^2  = (-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4
		# y - (Fy+Rd)/2 = +/-sqrt[(-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4]
		# [7] y = +/-sqrt[(-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4] + (Fy+Rd)/2
		# We are only interested in the bottom half of the circle:
		# [8] y = -sqrt[(-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4] + (Fy+Rd)/2
		# Eq [8] must intersect Eq [1]:
		# -sqrt[(-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4] + (Fy+Rd)/2 = -sqrt(Rf^2 - (x-Fx)^2) + Fy
		# Rearrange to find the zero:
		# [9] -sqrt[(-x^2 + Fx*x)/2 - Rd*Fy + (Fy+Rd)^2/4] + (Fy+Rd)/2 +
		#   sqrt(Rf^2 - (x-Fx)^2) - Fy = 0

		G = _gear
		Rd = _Rd

		# find center of fillet circle
		Fx = (G.Rr + G.Rf)*sin(pi/G.N)
		Fy = (G.Rr + G.Rf)*cos(pi/G.N)

		def func(x):
			# Eq [9]
			return real(-sqrt( (-(x**2) + Fx*x)/2 - Rd*Fy + ((Fy+Rd)**2)/4) + (Fy+Rd)/2 +
			    sqrt(G.Rf**2 - (x-Fx)**2) - Fy)

		# starting point is left side of fillet circle (centerpoint minus radius)
		initialGuess = Fx-G.Rf

		x = fsolve(func, initialGuess)
		y = real(-sqrt(G.Rf**2 - (x-Fx)**2) + Fy)	# Eq [1]
		a = (y-Rd)/(x**2)						# Eq [3]

		## Results
		# get rid of any complex number residuals
		x = real(x)
		y = real(y)
		a = real(a)

		return x, y, a

	def stress(self, _torque):
		"""Calculates the stress on the gear teeth.

		Args
		----
		torque (float) : torque, Nm
		"""

		FW = min(self.G1.FW, self.G2.FW)

		# Highest point of single tooth contact
		phi_hp1 = arccos(self.G1.Rb/self.G1.Rhp)
		theta_hp1 = (self.G1.tts/(2*self.G1.Rs)) + invF(self.OPA) - invF(phi_hp1)
		Rhp1_x = self.G1.Rhp*sin(theta_hp1)
		Rhp1_y = self.G1.Rhp*cos(theta_hp1)

		phi_hp2 = arccos(self.G2.Rb/self.G2.Rhp)
		theta_hp2 = (self.G2.tts/(2*self.G2.Rs)) + invF(self.OPA) - invF(phi_hp2)
		Rhp2_x = self.G2.Rhp*sin(theta_hp2)
		Rhp2_y = self.G2.Rhp*cos(theta_hp2)

		# Lewis parabola
		Rd1 = self.G1.Rb / cos(phi_hp1 - theta_hp1)
		[x1_Lewis, y1_Lewis, a1_Lewis] = self.lewisParabola(self.G1, Rd1)

		Rd2 = self.G2.Rb / cos(phi_hp2 - theta_hp2)
		[x2_Lewis, y2_Lewis, a2_Lewis] = self.lewisParabola(self.G2, Rd2)

		# Lewis parabola dimensions
		tt_LP1 = 2*x1_Lewis   # tooth thickness at critical section
		tt_LP2 = 2*x2_Lewis
		h_LP1 = Rd1 - y1_Lewis # height of Lewis parabola
		h_LP2 = Rd2 - y2_Lewis

		# Stress concentration factor Kf
		# from GOIG 11.24 - 11.26
		# Note that GOIG uses degrees while AGMA 908 uses radians
		k1 = 0.3054 - 0.00489*self.OPA_deg - 0.000069*self.OPA_deg**2
		k2 = 0.3620 - 0.01268*self.OPA_deg + 0.000104*self.OPA_deg**2
		k3 = 0.2934 + 0.00609*self.OPA_deg + 0.000087*self.OPA_deg**2

		Kf1 = k1 + ( (tt_LP1/self.G1.Rf)**k2 ) * ( (tt_LP1/h_LP1)**k3 )
		Kf2 = k1 + ( (tt_LP2/self.G2.Rf)**k2 ) * ( (tt_LP2/h_LP2)**k3 )

		# # convert torque to force through HPSTC tangent to base circle
		w1 = _torque / (self.G1.Rb * FW)
		w2 = w1

		# # angle between force normal direction and perpendicular to tooth
		# # centerline
		gamma1 = phi_hp1 - theta_hp1
		gamma2 = phi_hp2 - theta_hp2

		stress1 = (w1/self.G1.mod) * cos(gamma1) * (Kf1*( ((1.5*self.G1.mod*h_LP1)/ x1_Lewis**2) -
					(0.5*self.G1.mod*tan(gamma1))/x1_Lewis ) )
		stress2 = (w2/self.G1.mod) * cos(gamma2) * (Kf2*( ((1.5*self.G1.mod*h_LP2)/ x2_Lewis**2) -
					(0.5*self.G1.mod*tan(gamma2))/x2_Lewis ) )

		print("Stress 1 (MPa): ", stress1)
		print("Stress 2 (MPa): ", stress2)