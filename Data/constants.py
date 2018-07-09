# Conversion factors
toBohr = 1.8897161646320724
toAng = 1.0/toBohr

# Atomic structure parameters
nElectrons = {'H':1,'HE':2,'LI':3,'BE':4,'B':5,'C':6,'N':7,'O':8,'F':9,'NE':10,
              'NA':11,'MG':12,'AL':13,'SI':14,'P':15,'S':16,'CL':17,'AR':18,
              'K':19,'CA':20,'SC':21,'TI':22,'V':23,'CR':24,'MN':25,'FE':26,'CO':27,
              'NI':28,'CU':29,'ZN':30,'GA':31,'GE':32,'AS':33,'SE':34,'BR':25,'KR':36}
nCoreOrbitals = {'H':0,'HE':0,'LI':1,'BE':1,'B':1,'C':1,'N':1,'O':1,'F':1,'NE':1,
                 'NA':5,'MG':5,'AL':5,'SI':5,'P':5,'S':5,'CL':5,'AR':5,
                 'K':9,'CA':9,'SC':9,'TI':9,'V':9,'CR':9,'MN':9,'FE':9,'CO':9,
                 'NI':9,'CU':9,'ZN':9,'GA':9,'GE':9,'AS':9,'SE':9,'BR':9,'KR':9}
nOrbitals = {'H':1,'HE':1,'LI':5,'BE':5,'B':5,'C':5,'N':5,'O':5,'F':5,'NE':5,
             'NA':9,'MG':9,'AL':9,'SI':9,'P':9,'S':9,'CL':9,'AR':9,
             'K':13,'CA':13,'SC':13,'TI':18,'V':18,'CR':18,'MN':18,'FE':18,'CO':18,
             'NI':18,'CU':18,'ZN':18,'GA':18,'GE':18,'AS':18,'SE':18,'BR':18,'KR':18}
atomicMultiplicity = {'H':2,'HE':1,'LI':2,'BE':1,'B':2,'C':3,'N':4,'O':3,'F':2,'NE':1,
                      'NA':2,'MG':1,'AL':2,'SI':3,'P':4,'S':3,'CL':2,'AR':1,
                      'K':2,'CA':1,'SC':2,'TI':3,'V':4,'CR':5,'MN':4,'FE':3,'CO':2,
                      'NI':1,'CU':2,'ZN':1,'GA':2,'GE':3,'AS':4,'SE':3,'BR':2,'KR':1}

# Angular momentum quantum numbers and set sizes
nAngMomCart = {}; nAngMomSpher = {};
for i in range(1,100):
  nAngMomCart[i-1] = (i*i+i)/2
for i in range(0,99):
  nAngMomSpher[i] = 2*i+1

# Settings and thresholds
integral_threshold = 1.e-8
energy_convergence = 1.e-6
density_convergence = 1.e-6
linear_dependence = 1.e-6
DIIS_convergence = 1.e-5
DIIS_max_condition = 1e6
NOCI_thresh = 1.e-10


