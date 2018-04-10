import numpy as np
import pdb

writeit = True
prefix = 'radiation_z1p724_uvbslope'
slopes = ['-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25', '0.5', '0.75', '1.0']

prenams = ['UVBslope', '[C/H]', 'Hescl', 'nH', 'log NHI']

cd_nam = ['H', 'H+',
          'He', 'He+', 'He+2',
          'C', 'C+', 'C+2', 'C+3',
          'O', 'O+', 'O+2', 'O+3', 'O+4', 'O+5', 'O+6',
          'Mg', 'Mg+', 'Mg+2',
          'Si', 'Si+', 'Si+2', 'Si+3', 'Si+4']

cldycols = ['H', 'H+', 'H-', 'H2', 'H2*', 'H2+', 'H3+', 'He', 'He+', 'He+2', 'HeH+', 'Li', 'Li+', 'Li+2', 'Li+3', 'Li-',
            'LiH', 'LiH+', 'Be', 'Be+', 'Be+2', 'Be+3', 'Be+4', 'B', 'B+', 'B+2', 'B+3', 'B+4', 'B+5', 'C', 'C+', 'C+2',
            'C+3', 'C+4', 'C+5', 'C+6', 'CH', 'CH+', 'CH2', 'CH2+', 'CH3', 'CH3+', 'CH4', 'CH4+', 'CH5+', 'C2', 'C2+',
            'C2H', 'C2H+', 'C2H2', 'C2H2+', 'C2H3+', 'C3', 'C3+', 'C3H', 'C3H+', 'N', 'N+', 'N+2', 'N+3', 'N+4', 'N+5',
            'N+6', 'N+7', 'NH', 'NH+', 'NH2', 'NH2+', 'NH3', 'NH3+', 'NH4+', 'CN', 'CN+', 'HCN', 'HCN+', 'HNC', 'HCNH+',
            'HC3N', 'N2', 'N2+', 'N2H+', 'O', 'O+', 'O+2', 'O+3', 'O+4', 'O+5', 'O+6', 'O+7', 'O+8', 'OH', 'OH+', 'H2O',
            'H2O+', 'H3O+', 'CO', 'CO+', 'HCO+', 'H2CO', 'CH3OH', '^13CO', 'NO', 'NO+', 'HNO', 'HNO+', 'OCN', 'OCN+',
            'N2O', 'O2', 'O2+', 'NO2', 'NO2+', 'F', 'F+', 'F+2', 'F+3', 'F+4', 'F+5', 'F+6', 'F+7', 'F+8', 'F+9', 'Ne',
            'Ne+', 'Ne+10', 'Ne+2', 'Ne+3', 'Ne+4', 'Ne+5', 'Ne+6', 'Ne+7', 'Ne+8', 'Ne+9', 'NeH+', 'Na', 'Na+',
            'Na+10', 'Na+11', 'Na+2', 'Na+3', 'Na+4', 'Na+5', 'Na+6', 'Na+7', 'Na+8', 'Na+9', 'Mg', 'Mg+', 'Mg+10',
            'Mg+11', 'Mg+12', 'Mg+2', 'Mg+3', 'Mg+4', 'Mg+5', 'Mg+6', 'Mg+7', 'Mg+8', 'Mg+9', 'Al', 'Al+', 'Al+10',
            'Al+11', 'Al+12', 'Al+13', 'Al+2', 'Al+3', 'Al+4', 'Al+5', 'Al+6', 'Al+7', 'Al+8', 'Al+9', 'Si', 'Si+',
            'Si+10', 'Si+11', 'Si+12', 'Si+13', 'Si+14', 'Si+2', 'Si+3', 'Si+4', 'Si+5', 'Si+6', 'Si+7', 'Si+8', 'Si+9',
            'SiH', 'SiH2+', 'SiC2', 'SiN', 'SiN+', 'SiO', 'SiO+', 'SiOH+', 'P', 'P+', 'P+10', 'P+11', 'P+12', 'P+13',
            'P+14', 'P+15', 'P+2', 'P+3', 'P+4', 'P+5', 'P+6', 'P+7', 'P+8', 'P+9', 'S', 'S+', 'S+10', 'S+11', 'S+12',
            'S+13', 'S+14', 'S+15', 'S+16', 'S+2', 'S+3', 'S+4', 'S+5', 'S+6', 'S+7', 'S+8', 'S+9', 'HS', 'HS+', 'CS',
            'CS+', 'HCS+', 'NS', 'NS+', 'SO', 'SO+', 'OCS', 'OCS+', 'SO2', 'S2', 'S2+', 'Cl', 'Cl+', 'Cl+10', 'Cl+11',
            'Cl+12', 'Cl+13', 'Cl+14', 'Cl+15', 'Cl+16', 'Cl+17', 'Cl+2', 'Cl+3', 'Cl+4', 'Cl+5', 'Cl+6', 'Cl+7',
            'Cl+8', 'Cl+9', 'HCl', 'HCl+', 'H2Cl+', 'CCl', 'CCl+', 'H2CCl+', 'ClO', 'ClO+', 'Ar', 'Ar+', 'Ar+10',
            'Ar+11', 'Ar+12', 'Ar+13', 'Ar+14', 'Ar+15', 'Ar+16', 'Ar+17', 'Ar+18', 'Ar+2', 'Ar+3', 'Ar+4', 'Ar+5',
            'Ar+6', 'Ar+7', 'Ar+8', 'Ar+9', 'ArH+', 'K', 'K+', 'K+10', 'K+11', 'K+12', 'K+13', 'K+14', 'K+15', 'K+16',
            'K+17', 'K+18', 'K+19', 'K+2', 'K+3', 'K+4', 'K+5', 'K+6', 'K+7', 'K+8', 'K+9', 'Ca', 'Ca+', 'Ca+10',
            'Ca+11', 'Ca+12', 'Ca+13', 'Ca+14', 'Ca+15', 'Ca+16', 'Ca+17', 'Ca+18', 'Ca+19', 'Ca+2', 'Ca+20', 'Ca+3',
            'Ca+4', 'Ca+5', 'Ca+6', 'Ca+7', 'Ca+8', 'Ca+9', 'Sc', 'Sc+', 'Sc+10', 'Sc+11', 'Sc+12', 'Sc+13', 'Sc+14',
            'Sc+15', 'Sc+16', 'Sc+17', 'Sc+18', 'Sc+19', 'Sc+2', 'Sc+20', 'Sc+21', 'Sc+3', 'Sc+4', 'Sc+5', 'Sc+6',
            'Sc+7', 'Sc+8', 'Sc+9', 'Ti', 'Ti+', 'Ti+10', 'Ti+11', 'Ti+12', 'Ti+13', 'Ti+14', 'Ti+15', 'Ti+16', 'Ti+17',
            'Ti+18', 'Ti+19', 'Ti+2', 'Ti+20', 'Ti+21', 'Ti+22', 'Ti+3', 'Ti+4', 'Ti+5', 'Ti+6', 'Ti+7', 'Ti+8', 'Ti+9',
            'TiH', 'TiH+', 'TiH2', 'TiH2+', 'TiC', 'TiC+', 'HCTi', 'HCTi+', 'TiC2', 'TiC2+', 'TiN', 'TiN+', 'HNTi',
            'HNTi+', 'TiNC', 'TiNC+', 'TiO', 'TiO+', 'TiOH+', 'TiO2', 'TiF+', 'TiS', 'TiS+', 'HTiS', 'HTiS+', 'V', 'V+',
            'V+10', 'V+11', 'V+12', 'V+13', 'V+14', 'V+15', 'V+16', 'V+17', 'V+18', 'V+19', 'V+2', 'V+20', 'V+21',
            'V+22', 'V+23', 'V+3', 'V+4', 'V+5', 'V+6', 'V+7', 'V+8', 'V+9', 'Cr', 'Cr+', 'Cr+10', 'Cr+11', 'Cr+12',
            'Cr+13', 'Cr+14', 'Cr+15', 'Cr+16', 'Cr+17', 'Cr+18', 'Cr+19', 'Cr+2', 'Cr+20', 'Cr+21', 'Cr+22', 'Cr+23',
            'Cr+24', 'Cr+3', 'Cr+4', 'Cr+5', 'Cr+6', 'Cr+7', 'Cr+8', 'Cr+9', 'Mn', 'Mn+', 'Mn+10', 'Mn+11', 'Mn+12',
            'Mn+13', 'Mn+14', 'Mn+15', 'Mn+16', 'Mn+17', 'Mn+18', 'Mn+19', 'Mn+2', 'Mn+20', 'Mn+21', 'Mn+22', 'Mn+23',
            'Mn+24', 'Mn+25', 'Mn+3', 'Mn+4', 'Mn+5', 'Mn+6', 'Mn+7', 'Mn+8', 'Mn+9', 'Fe', 'Fe+', 'Fe+10', 'Fe+11',
            'Fe+12', 'Fe+13', 'Fe+14', 'Fe+15', 'Fe+16', 'Fe+17', 'Fe+18', 'Fe+19', 'Fe+2', 'Fe+20', 'Fe+21', 'Fe+22',
            'Fe+23', 'Fe+24', 'Fe+25', 'Fe+26', 'Fe+3', 'Fe+4', 'Fe+5', 'Fe+6', 'Fe+7', 'Fe+8', 'Fe+9', 'Co', 'Co+',
            'Co+10', 'Co+11', 'Co+12', 'Co+13', 'Co+14', 'Co+15', 'Co+16', 'Co+17', 'Co+18', 'Co+19', 'Co+2', 'Co+20',
            'Co+21', 'Co+22', 'Co+23', 'Co+24', 'Co+25', 'Co+26', 'Co+27', 'Co+3', 'Co+4', 'Co+5', 'Co+6', 'Co+7',
            'Co+8', 'Co+9', 'Ni', 'Ni+', 'Ni+10', 'Ni+11', 'Ni+12', 'Ni+13', 'Ni+14', 'Ni+15', 'Ni+16', 'Ni+17',
            'Ni+18', 'Ni+19', 'Ni+2', 'Ni+20', 'Ni+21', 'Ni+22', 'Ni+23', 'Ni+24', 'Ni+25', 'Ni+26', 'Ni+27', 'Ni+28',
            'Ni+3', 'Ni+4', 'Ni+5', 'Ni+6', 'Ni+7', 'Ni+8', 'Ni+9', 'Cu', 'Cu+', 'Cu+10', 'Cu+11', 'Cu+12', 'Cu+13',
            'Cu+14', 'Cu+15', 'Cu+16', 'Cu+17', 'Cu+18', 'Cu+19', 'Cu+2', 'Cu+20', 'Cu+21', 'Cu+22', 'Cu+23', 'Cu+24',
            'Cu+25', 'Cu+26', 'Cu+27', 'Cu+28', 'Cu+29', 'Cu+3', 'Cu+4', 'Cu+5', 'Cu+6', 'Cu+7', 'Cu+8', 'Cu+9', 'Zn',
            'Zn+', 'Zn+10', 'Zn+11', 'Zn+12', 'Zn+13', 'Zn+14', 'Zn+15', 'Zn+16', 'Zn+17', 'Zn+18', 'Zn+19', 'Zn+2',
            'Zn+20', 'Zn+21', 'Zn+22', 'Zn+23', 'Zn+24', 'Zn+25', 'Zn+26', 'Zn+27', 'Zn+28', 'Zn+29', 'Zn+3', 'Zn+30',
            'Zn+4', 'Zn+5', 'Zn+6', 'Zn+7', 'Zn+8', 'Zn+9', 'CRP', 'CRPHOT', 'PHOTON', 'e-', 'grn']

cd_idx = []
for i in cd_nam:
    for j in range(len(cldycols)):
        if i == cldycols[j]:
            cd_idx.append(j)
            break

stchdata = None
for slope in slopes:
    fname = '{0:s}{1:s}'.format(prefix, slope)
    print('Preparing to stitch {0:s}'.format(fname))
    metals, hescl, hden, NHI = np.loadtxt(fname + '.grd', unpack=True, usecols=(6, 7, 8, 9))
    slps = np.ones_like(metals) * float(slope)

    print('Loading...')
    data = np.loadtxt(fname+'.clm', usecols=tuple(cd_idx))
    print('loaded')

    print('Appending...')
    if stchdata is None:
        stchdata = np.append(np.vstack((slps, metals, hescl, hden, NHI)).T, data, axis=1)
    else:
        subdata = np.append(np.vstack((slps, metals, hescl, hden, NHI)).T, data, axis=1)
        stchdata = np.append(stchdata, subdata, axis=0)
    print('Appended')


if writeit:
    print('Saving')
    np.save(prefix+'_data', stchdata)
    print('Complete')
else:
    # Print out some random location
    while True:
        idx = stchdata.shape[0] * np.random.uniform(0.0, 1.0)
        if stchdata[int(idx), 0] == 0.0:
            break
    for i in range(stchdata.shape[1]):
        if i < len(prenams):
            print(prenams[i]+" = ", stchdata[int(idx), i])
        else:
            print(cd_nam[i-len(prenams)] + " = ", np.log10(stchdata[int(idx), i]))
