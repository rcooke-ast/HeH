import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import argparse 

#pick range of alpha
allalpha=np.arange(-2,2.5,0.5)

#load the uvb only case Hz, nu 4pi Jnu 
#load UVB only
uvbonly=np.loadtxt('radiation_z1p724_uvbonly.cnt',dtype={'names': ('nu', 'int', 'phot'),'formats': ('f8', 'f8', 'f8')})

#go to log 10 Jnu
uvbonly['int']=np.log10(uvbonly['int']/(4*np.pi*uvbonly['nu']))

#build interpolation function 
uvbfunc=interp1d(uvbonly['nu'],uvbonly['int'])

#from energy to frequency dividing by h
Rydnu=13.605698/4.1357e-15 #1Rydberg
Ryd10nu=10*Rydnu #10 Rydberg

for alpha in allalpha:

    #modify as in eq B5 of Crighton et al. 2015
    #Update equation to fix bug in NC2015
    uvbslope=np.copy(uvbonly)

    midfreq=np.where((uvbslope['nu'] >= Rydnu) & (uvbslope['nu'] <= Ryd10nu))
    uvbslope['int'][midfreq]=uvbonly['int'][midfreq]+alpha*np.log10(uvbonly['nu'][midfreq]/Rydnu)
    
    highfreq=np.where(uvbslope['nu'] > Ryd10nu)
    uvbslope['int'][highfreq]=uvbonly['int'][highfreq]+alpha*np.log10(Ryd10nu/Rydnu)
    
    plt.plot(np.log10(uvbslope['nu']),uvbslope['int'],label='UVBSlope {}'.format(alpha))

    #now write the updated UVB continuum input file
    cntfile=file('radiation_z1p724_uvbslope{}.in'.format(alpha),'w')

    #copy template
    template=file('radiation_z1p724_uvbslope_template.in')
    for line in template:
        cntfile.write(line)
        if('ModifiedUVB' in line):
            #sample new UVB 1 every 4 lines 
            sample=0
            for ii,freq in enumerate(uvbslope['nu']):
                if(ii == 0):
                    cntfile.write('interpolate ({},{})\n'.format(np.log10(uvbslope['nu'][ii]),uvbslope['int'][ii]))
                else:
                    #write with full sampling in frequency
                    if((uvbslope['nu'][ii] > Rydnu) & (uvbslope['nu'][ii] < Ryd10nu)):
                        sample=0
                        cntfile.write('continue ({},{})\n'.format(np.log10(uvbslope['nu'][ii]),uvbslope['int'][ii]))
                    else:
                        #write only 1 in 6
                        sample=sample+1
                        if((sample > 6) & (np.isfinite(uvbslope['int'][ii]))):
                            cntfile.write('continue ({},{})\n'.format(np.log10(uvbslope['nu'][ii]),uvbslope['int'][ii]))
                            sample=0
                

    #add save files
    cntfile.write('save grid "radiation_z1p724_uvbslope{}.grd"\n'.format(alpha))
    cntfile.write('save last species ALL column densities "radiation_z1p724_uvbslope{}.clm"\n'.format(alpha))
    cntfile.write('save last ionization means "radiation_z1p724_uvbslope{}.ion"\n'.format(alpha))
                   
    cntfile.close()
    template.close()

#display    
plt.plot(np.log10(uvbonly['nu']),uvbonly['int'],label='UVB',linestyle=':',color='black')
plt.axvline(x=np.log10(Rydnu),linestyle='--',color='black')
plt.axvline(x=np.log10(Ryd10nu),linestyle='--',color='black')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$\nu$ 4 $\pi$ J$_\nu$ [erg cm$^-2$ s$^-1$]')    
plt.legend()
plt.show()

