import numpy as np
import sys
import pdb
from copy import deepcopy
import copy as cp
import astropy.units as u
import mpl_scatter_density
from pylab import *

import seaborn as sns
plt.style.use('seaborn')

sys.path.append('/Users/anh5866/Desktop/Coding')
import PyXFocus.sources as sources
import PyXFocus.transformations as trans
import PyXFocus.surfaces as surfaces
import PyXFocus.analyses as analyses
import PyXFocus.conicsolve as conic

import OGRE.ogre_routines_alexplay as ogre



class OGRERays:
    def __init__(self, PyXFocusRays, wave, order):
        self.opd = PyXFocusRays[0]
        self.x = PyXFocusRays[1]
        self.y = PyXFocusRays[2]
        self.z = PyXFocusRays[3]
        self.vx = PyXFocusRays[4]
        self.vy = PyXFocusRays[5]
        self.vz = PyXFocusRays[6]
        self.nx = PyXFocusRays[7]
        self.ny = PyXFocusRays[8]
        self.nz = PyXFocusRays[9]
        self.wave = wave
        self.order = order
        self.index = np.arange(len(PyXFocusRays[0]))
        self.weight = np.ones(len(PyXFocusRays[0]))
        
    def set_prays(self,PyXFocusRays, ind = None):
        if ind is not None:
            self.opd[ind] = PyXFocusRays[0]
            self.x[ind] = PyXFocusRays[1]
            self.y[ind] = PyXFocusRays[2]
            self.z[ind] = PyXFocusRays[3]
            self.vx[ind] = PyXFocusRays[4]
            self.vy[ind] = PyXFocusRays[5]
            self.vz[ind] = PyXFocusRays[6]
            self.nx[ind] = PyXFocusRays[7]
            self.ny[ind] = PyXFocusRays[8]
            self.nz[ind] = PyXFocusRays[9]
        else:
            self.opd = PyXFocusRays[0]
            self.x = PyXFocusRays[1]
            self.y = PyXFocusRays[2]
            self.z = PyXFocusRays[3]
            self.vx = PyXFocusRays[4]
            self.vy = PyXFocusRays[5]
            self.vz = PyXFocusRays[6]
            self.nx = PyXFocusRays[7]
            self.ny = PyXFocusRays[8]
            self.nz = PyXFocusRays[9]
            
    def yield_prays(self, ind = None):
        if ind is not None:
            return [self.opd[ind],self.x[ind],self.y[ind],self.z[ind],self.vx[ind],self.vy[ind],self.vz[ind],self.nx[ind],self.ny[ind],self.nz[ind]]
        else:
            return [self.opd,self.x,self.y,self.z,self.vx,self.vy,self.vz,self.nx,self.ny,self.nz]
        
        #returns parameters 

    def yield_object_indices(self, ind):
        new_object = cp.deepcopy(self)
        
        for key in self.__dict__.keys():
            attr = self.__dict__[key]
            try:
                new_object.__dict__[key] = attr[ind]
            except (TypeError, IndexError):
                #Not indexable, keep as-is
                new_object.__dict__[key] = attr
                
        return new_object