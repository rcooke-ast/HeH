import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import trapz

#load the uvb only case Hz, nu 4pi Jnu 
#uvbonly=Table.read('radiation_z1p724_uvbonly.cnt',format='ascii',header_start=1,data_start=2)

#load UVB only
uvbonly=np.loadtxt('radiation_z1p724_uvbonly.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})
uvbqso=np.loadtxt('radiation_z1p724_uvbplusqso.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})
uvbgal=np.loadtxt('radiation_z1p724_uvbplusgal.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})

#function
uvbfunc=interp1d(uvbonly['nu'],uvbonly['int']/uvbonly['nu'])
uvbgalfunc=interp1d(uvbgal['nu'],uvbgal['int']/uvbgal['nu'])
uvbqsofunc=interp1d(uvbqso['nu'],uvbqso['int']/uvbqso['nu'])

#integrate 
Rydnu=13.605698/4.1357e-15 
UVBint,UVBerr=quad(uvbfunc,Rydnu,2e22)
print('Log (4piJnu UVB >1Ry)',np.log10(UVBint))
UVBqsoint,UVBqsoerr=quad(uvbqsofunc,Rydnu,2e22)
print('Log (4piJnu QSO >1Ry)',np.log10(UVBqsoint))

#slice array
highv=(np.where(uvbgal['nu'] > Rydnu))[0]
intgr=np.array(uvbgal['int'][highv]/uvbgal['nu'][highv])
intx=np.array(uvbgal['nu'][highv])
print(intgr.shape,intx.shape)
output=trapz(intgr,x=intx)
print('Log (4piJnu GAL >1Ry simps)',np.log10(output))

#show
plt.plot(np.log10(uvbonly['nu']),np.log10(uvbonly['int']),label='UVB')
plt.plot(np.log10(uvbgal['nu']),np.log10(uvbgal['int']),label='Galaxy')
plt.plot(np.log10(uvbqso['nu']),np.log10(uvbqso['int']),label='AGN')
plt.axvline(x=np.log10(Rydnu),linestyle='--',color='black')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$\nu$ 4 $\pi$ J$_\nu$ [erg cm$^-2$ s$^-1$]')
plt.show()

