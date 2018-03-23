import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

#load the uvb only case Hz, nu 4pi Jnu 
#uvbonly=Table.read('radiation_z1p724_uvbonly.cnt',format='ascii',header_start=1,data_start=2)

#load UVB only
uvbonly=np.loadtxt('radiation_z1p724_uvbonly.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})
uvbqso=np.loadtxt('radiation_z1p724_uvbplusqso.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})

#function
uvbfunc=interp1d(uvbonly['nu'],uvbonly['int']/uvbonly['nu'])
uvbqsofunc=interp1d(uvbqso['nu'],uvbqso['int']/uvbqso['nu'])

#integrate 
Rydnu=13.605698/4.1357e-15 
UVBint,UVBerr=quad(uvbfunc,Rydnu,2e22)
print('Log (4piJnu >1Ry)',np.log10(UVBint))
UVBqsoint,UVBqsoerr=quad(uvbqsofunc,Rydnu,2e22)
print('Log (4piJnu qso >1Ry)',np.log10(UVBqsoint))

#show
plt.plot(np.log10(uvbonly['nu']),np.log10(uvbonly['int']))
plt.plot(np.log10(uvbqso['nu']),np.log10(uvbqso['int']))
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$\nu$ 4 $\pi$ J$_\nu$ [erg cm$^-2$ s$^-1$]')
plt.show()

