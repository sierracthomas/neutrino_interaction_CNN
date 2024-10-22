from math import log, isnan
import numpy as np
"""
HEP coherent system of units

The basic units are :
    millimeter              (millimeter)
    nanosecond              (nanosecond)
    Mega electron Volt      (MeV)
    positron charge         (e)
    degree Kelvin           (kelvin)
    the amount of substance (mole)
    luminous intensity      (candela)
    radian                  (radian)
    steradian               (steradian)

"""

# Length [L]

euro = 1.
millimeter = 1.
millimeter2 = millimeter * millimeter
millimeter3 = millimeter * millimeter2

centimeter = 10. * millimeter
centimeter2 = centimeter * centimeter
centimeter3 = centimeter * centimeter2

decimeter = 100. * millimeter
decimeter2 = decimeter * decimeter
decimeter3 = decimeter * decimeter2
liter = decimeter3
l = liter
ml  = 1e-3 * l
mul = 1e-6 * l
nl  = 1e-9 * l
pl  = 1e-12 * l

meter = 1000. * millimeter
meter2 = meter * meter
meter3 = meter * meter2

kilometer = 1000. * meter
kilometer2 = kilometer * kilometer
kilometer3 = kilometer * kilometer2

micrometer = 1.e-6 * meter
nanometer = 1.e-9 * meter
angstrom = 1.e-10 * meter
fermi = 1.e-15 * meter

nm = nanometer
mum = micrometer

micron = micrometer
micron2 = micrometer * micrometer
micron3 = micron2 * micrometer

barn = 1.e-28 * meter2
millibarn = 1.e-3 * barn
microbarn = 1.e-6 * barn
nanobarn = 1.e-9 * barn
picobarn = 1.e-12 * barn

# symbols
mm = millimeter
mm2 = millimeter2
mm3 = millimeter3

cm = centimeter
cm2 = centimeter2
cm3 = centimeter3

m = meter
m2 = meter2
m3 = meter3

km = kilometer
km2 = kilometer2
km3 = kilometer3

ft = 30.48 * cm

# Angle

radian = 1.
milliradian = 1.e-3 * radian
degree = (3.14159265358979323846/180.0) * radian

steradian = 1.

# symbols
rad = radian
mrad = milliradian
sr = steradian
deg = degree

# Time [T]

nanosecond = 1.
second = 1.e+9 * nanosecond
millisecond = 1.e-3 * second
microsecond = 1.e-6 * second
picosecond = 1.e-12 * second
femtosecond = 1.e-15 * second
year = 3.1536e+7 * second
day = 864e2 * second
minute = 60 * second
hour = 60 * minute

s = second
ms = millisecond
ps = picosecond
fs = femtosecond
mus = microsecond
ns = nanosecond

hertz = 1./second
kilohertz = 1.e+3 * hertz
megahertz = 1.e+6 * hertz
gigahertz = 1.e+6 * hertz

MHZ = megahertz
kHZ = kilohertz
kHz = kHZ
GHZ = gigahertz

# Electric charge [Q]

e = 1. # electron charge
e_SI = -1.60217733e-19 # electron charge in coulomb
coulomb = e/e_SI # coulomb = 6.24150 e+18 * e

# Energy [E]

megaelectronvolt = 1.
electronvolt = 1.e-6 * megaelectronvolt
milielectronvolt = 1.e-3 * electronvolt
kiloelectronvolt = 1.e-3 * megaelectronvolt
gigaelectronvolt = 1.e+3 * megaelectronvolt
teraelectronvolt = 1.e+6 * megaelectronvolt
petaelectronvolt = 1.e+9 * megaelectronvolt

meV = milielectronvolt
eV = electronvolt
keV = kiloelectronvolt
MeV = megaelectronvolt
GeV = gigaelectronvolt
TeV = teraelectronvolt
PeV = petaelectronvolt

eV2 = eV*eV

joule = electronvolt/e_SI # joule = 6.24150 e+12 * MeV
J     = joule
milijoule = 1e-3 * joule
microjoule = 1e-6 * joule
nanojoule = 1e-9 * joule
picojoule = 1e-12 * joule
femtojoule = 1e-15 * joule
mJ  = milijoule
muJ = microjoule
nJ  = nanojoule
pJ  = picojoule
fJ  = femtojoule

# Mass [E][T^2][L^-2]

kilogram = joule * second * second / meter2
gram = 1.e-3 * kilogram
milligram = 1.e-3 * gram
ton = 1.e+3 * kilogram
kiloton = 1.e+3 * ton

# symbols
kg = kilogram
g = gram
mg = milligram

# Power [E][T^-1]

watt = joule/second # watt = 6.24150 e+3 * MeV/ns
W    = watt
milliwatt = 1E-3 * watt
microwatt = 1E-6 * watt
mW = milliwatt
muW = microwatt

# Force [E][L^-1]

newton = joule/meter  # newton = 6.24150 e+9 * MeV/mm

# Pressure [E][L^-3]

hep_pascal = newton / m2 # pascal = 6.24150 e+3 * MeV/mm3
pascal = hep_pascal
Pa = pascal
kPa = 1000 * Pa
MPa = 1e+6 * Pa
GPa = 1e+9 * Pa
bar = 100000 * pascal # bar = 6.24150 e+8 * MeV/mm3
milibar = 1e-3 * bar

atmosphere = 101325 * pascal # atm = 6.32420 e+8 * MeV/mm3

denier = gram / (9000 * meter)

# Electric current [Q][T^-1]

ampere = coulomb/second # ampere = 6.24150 e+9 * e/ns
milliampere = 1.e-3 * ampere
microampere = 1.e-6 * ampere
nanoampere = 1.e-9 * ampere
mA = milliampere
muA = microampere
nA = nanoampere

# Electric potential [E][Q^-1]

megavolt = megaelectronvolt / e
kilovolt = 1.e-3 * megavolt
volt = 1.e-6 * megavolt
millivolt = 1.e-3 * volt

V = volt
mV = millivolt
kV = kilovolt
MV = megavolt

# Electric resistance [E][T][Q^-2]

ohm = volt / ampere # ohm = 1.60217e-16*(MeV/e)/(e/ns)

# Electric capacitance [Q^2][E^-1]

farad = coulomb / volt # farad = 6.24150e+24 * e/Megavolt
millifarad = 1.e-3 * farad
microfarad = 1.e-6 * farad
nanofarad = 1.e-9 * farad
picofarad = 1.e-12 * farad

nF = nanofarad
pF = picofarad

# Magnetic Flux [T][E][Q^-1]

weber = volt * second # weber = 1000*megavolt*ns

# Magnetic Field [T][E][Q^-1][L^-2]

tesla = volt*second / meter2 # tesla = 0.001*megavolt*ns/mm2

gauss = 1.e-4 * tesla
kilogauss = 1.e-1 * tesla

# Inductance [T^2][E][Q^-2]

henry = weber / ampere # henry = 1.60217e-7*MeV*(ns/e)**2

# Temperature

kelvin = 1
K = kelvin

# Amount of substance

mole = 1
mol = mole
milimole    = 1E-3 * mole
micromole   = 1E-6 * mole
nanomole    = 1E-9 * mole
picomole    = 1E-12 * mole

# Activity [T^-1]

becquerel = 1 / second

curie = 3.7e+10 * becquerel

Bq = becquerel
mBq = 1e-3 * becquerel
muBq = 1e-6 * becquerel
kBq =  1e+3 * becquerel
MBq =  1e+6 * becquerel
cks = Bq/keV
U238ppb = Bq / 81
Th232ppb = Bq / 246

# Absorbed dose [L^2][T^-2]

gray = joule / kilogram

# Luminous intensity [I]

candela = 1

# Luminous flux [I]

lumen = candela * steradian

# Illuminance [I][L^-2]

lux = lumen / meter2

# Miscellaneous

perCent = 1e-2
perThousand = 1e-3
perMillion = 1e-6

pes = 1
adc = 1

def celsius(tKelvin):
    return tKelvin - 273.15



#: Detector temperature in K
TEMPERATURE = 87.17
#: Liquid argon density in :math:`g/cm^3`
LAR_DENSITY = 1.38 # g/cm^3
#: Electric field magnitude in :math:`kV/cm`
E_FIELD = 0.50 # kV/cm
#: Drift velocity in :math:`cm/\mu s`
V_DRIFT = 0.1648 # cm / us,
#: Electron lifetime in :math:`\mu s`
ELECTRON_LIFETIME = 2.2e3 # us,
#: Time sampling in :math:`\mu s`
TIME_SAMPLING = 0.1 # us
#: Drift time window in :math:`\mu s`
TIME_INTERVAL = (0, 200.) # us
#: Signal time window padding in :math:`\mu s`
TIME_PADDING = 10
#: Number of sampled points for each segment slice
SAMPLED_POINTS = 40
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
LONG_DIFF = 4.0e-6 # cm * cm / us
#: Transverse diffusion coefficient in :math:`cm^2/\mu s`
TRAN_DIFF = 8.8e-6 # cm * cm / us
#: Numpy array containing all the time ticks in the drift time window
TIME_TICKS = np.linspace(TIME_INTERVAL[0],
                         TIME_INTERVAL[1],
                         int(round(TIME_INTERVAL[1]-TIME_INTERVAL[0])/TIME_SAMPLING)+1)
#: Time window of current response in :math:`\mu s`
TIME_WINDOW = 8.9 # us
#: TPC drift length in :math:`cm`
DRIFT_LENGTH = 0
#: Time sampling in the pixel response file in :math:`\mu s`
RESPONSE_SAMPLING = 0.1
#: Spatial sampling in the pixel reponse file in :math:`cm`
RESPONSE_BIN_SIZE = 0.04434
#: Borders of each TPC volume in :math:`cm`
TPC_BORDERS = np.zeros((0, 3, 2))
#: TPC offsets wrt the origin in :math:`cm`
TPC_OFFSETS = np.zeros((0, 3, 2))
#: Pixel tile borders in :math:`cm`
TILE_BORDERS = np.zeros((2,2))
#: Default value for pixel_plane, to indicate out-of-bounds edep
DEFAULT_PLANE_INDEX = 0x0000BEEF
#: Total number of pixels
N_PIXELS = 0, 0
#: Number of pixels in each tile
N_PIXELS_PER_TILE = 0, 0
#: Dictionary between pixel ID and its position in the pixel array
PIXEL_CONNECTION_DICT = {}
#: Pixel pitch in :math:`cm`
PIXEL_PITCH = 0.4434
#: Tile position wrt the center of the anode in :math:`cm`
TILE_POSITIONS = {}
#: Tile orientations in each anode
TILE_ORIENTATIONS = {}
#: Map of tiles in each anode
TILE_MAP = ()
#: Association between chips and io channels
TILE_CHIP_TO_IO = {}
#: Association between modules and io groups
MODULE_TO_IO_GROUPS = {}
#: Association between modules and tpcs
MODULE_TO_TPCS = {}
TPC_TO_MODULE = {}


"""
Set physics constants
"""

## Physical params
#: Recombination :math:`\alpha` constant for the Box model
BOX_ALPHA = 0.93
#: Recombination :math:`\beta` value for the Box model in :math:`(kV/cm)(g/cm^2)/MeV`
BOX_BETA = 0.207 #0.3 (MeV/cm)^-1 * 1.383 (g/cm^3)* 0.5 (kV/cm), R. Acciarri et al JINST 8 (2013) P08005
#: Recombination :math:`A_b` value for the Birks Model
BIRKS_Ab = 0.800
#: Recombination :math:`k_b` value for the Birks Model in :math:`(kV/cm)(g/cm^2)/MeV`
BIRKS_kb = 0.0486 # g/cm2/MeV Amoruso, et al NIM A 523 (2004) 275
#: Electron charge in Coulomb
E_CHARGE = 1.602e-19
#: Average energy expended per ion pair in LAr in :math:`MeV` from Phys. Rev. A 10, 1452
W_ION = 23.6e-6

## Quenching parameters
BOX = 1
BIRKS = 2

#: Prescale factor analogous to ScintPreScale in LArSoft FIXME
SCINT_PRESCALE = 1
#: Ion + excitation work function in `MeV`
W_PH = 19.5e-6 # MeV

def quench(dEdx, dE, mode):
    
    dEdx = np.array(dEdx)
    dE = np.array(dE)

    recomb = 0
    if mode == BOX:
        # Baller, 2013 JINST 8 P08005
        csi = BOX_BETA * dEdx / (E_FIELD * LAR_DENSITY)
        logval = np.log(BOX_ALPHA + csi)/csi
        #recomb = max(0, np.log(BOX_ALPHA + csi)/csi)
        recomb = [np.max(0, i) for i in logval]
    elif mode == BIRKS:
        # Amoruso, et al NIM A 523 (2004) 275
        recomb = BIRKS_Ab / (1 + BIRKS_kb * dEdx / (E_FIELD * LAR_DENSITY))
    else:
        raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BIRKS'")


    n_electrons = recomb * dE / W_ION
    n_photons = (dE/W_PH - n_electrons) * SCINT_PRESCALE
    return n_electrons, n_photons