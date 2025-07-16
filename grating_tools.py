import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import cm
import sys
import pdb
from copy import deepcopy
import copy as cp
from tqdm import tqdm
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


class OGREGrating_prime:
    def __init__(self, hub, gammy):
        self.hub = hub * u.mm # Hub length
        d = 315.15 * u.nm # [nm] At 3300 mm from grating hub.
        d *= 3000 * u.mm / self.hub # Redefine to value at center of grating.

        self.d = d

        self.gammy = gammy * u.deg  # Graze angle. #not gamma #eta
        self.blaze = 0.575959 * u.rad  # Blaze angle. #33 degrees
        self.yaw = 0.98 * u.deg  # Yaw of grating.
        
        self.throw = self.hub / np.cos(self.gammy) 

        self.grat_length = 70 * u.mm
        self.grat_width = 63 * u.mm
        

    def calc_convergence(self, z, r_int):
        
        ave_r = np.mean(r_int)
        conv_ang = np.arctan(ave_r/z)

        return conv_ang


    def compute_throws(grating_centers):
        grating_centers = np.array(grating_centers)  # make sure it's a NumPy array
        throws = np.sqrt(grating_centers[0]**2 + grating_centers[1]**2 + grating_centers[2]**2)
        
        return throws


    def yield_grats(self, ind = None):
        if ind is not None:
            return [self.hub[ind],self.d[ind],self.gammy[ind],self.blaze[ind],self.yaw[ind],self.throw[ind],self.grat_length[ind],self.grat_width[ind]]
            
        else:
            return [self.hub,self.d,self.gammy,self.blaze,self.yaw,self.throw,self.grat_length,self.grat_width]
        


    def get_center_coords(self, r_bottom, r_top):
        r_mid = (r_top + r_bottom)/2
        alpha_4 = np.arcsin(r_mid/self.throw.value) 
        z_mid = self.throw.value * np.cos(alpha_4)

        x = 0
        y = r_mid
        z = z_mid

        cen_pos = np.array([x, y, z])

        theta = alpha_4 - math.radians(self.gammy.value)
        r_arc = self.throw.value * np.sin(math.radians(self.gammy.value))
        
        x_z = 0
        y_z = (2*self.throw.value * np.sin(math.radians(self.gammy.value))) * np.cos(theta)
        z_z = (2*self.throw.value * np.sin(math.radians(self.gammy.value))) * np.sin(theta)

        zero_pos = np.array([x_z, y_z, z_z])

        
        return cen_pos, zero_pos, r_arc


    def get_rowland(cen_pos, zero_pos):
    
        A = cen_pos
        B = zero_pos
        C = np.array([0,0,0]) #focus posiiton

        # Vectors
        AB = B - A
        AC = C - A

        # normal vector
        n = np.cross(AB, AC)
        n = n / np.linalg.norm(n)

        #midpoints
        mid_AB = (A + B) / 2
        mid_BC = (B + C) / 2

        #Directions
        perp_AB = np.cross(n, AB)
        perp_BC = np.cross(n, C - B)

        # system of equations

        M = np.column_stack((perp_AB, -perp_BC))
        rhs = mid_BC - mid_AB

        # Least-squares solution
        st, residuals, rank, s = np.linalg.lstsq(M, rhs, rcond=None)
        s_val = st[0]

        # Center of circle
        center = mid_AB + s_val * perp_AB

        # Radius
        radius = np.linalg.norm(center - A)
    
        return center, radius, n  # center, radius, normal vector



def stack_gratings(
    center_circle,       # center of the circle (3d vector)
    radius_circle,       # radius of the circle
    normal_vec,     # normal to the plane of the circle (unit vector)
    cen_pos,     # starting point (grating center A) on the circle
    spacing,      # desired arc-length spacing between gratings (in mm)
    grating_thickness,
    N             # number of gratings to generate (including cen_pos)
):

    focus_value = 3500 #mm

    u = (cen_pos - center_circle)
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vec, u)  # perpendicular in the plane

 
    delta_theta = (spacing + grating_thickness) / radius_circle
    indices = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
    theta_vals = indices * delta_theta

    grating_centers = [center_circle + radius_circle * (np.cos(theta) * u + np.sin(theta) * v) for theta in theta_vals]


    def compute_converge(grating_centers, focus_value): 
        grating_centers = np.array(grating_centers)  # ensure it's a NumPy array
        y_coords = grating_centers[:, 1]             # extract y column
        angles = np.arctan(y_coords / focus_value)

        return angles

    def compute_throws(grating_centers):
        grating_centers = np.array(grating_centers)  # make sure it's a NumPy array
        throws = np.linalg.norm(grating_centers[:, :3], axis=1)
        
        return throws

    convergence = compute_converge(grating_centers, focus_value)
    throw = compute_throws(grating_centers)
    
    return np.array(grating_centers), np.array(convergence), np.array(throw)



class OGREGrating:
    def __init__(self, grating_prime, single_grating_center, single_convergence, single_throw):
        self.hub = grating_prime.hub # Hub length
        self.d = grating_prime.d

        self.gammy = grating_prime.gammy # Graze angle. #not gamma #eta
        self.blaze = grating_prime.blaze  # Blaze angle. #33 degrees
        self.yaw = grating_prime.yaw # Yaw of grating.

        self.grat_length = 70 * u.mm
        self.grat_width = 63 * u.mm

        self.throw = single_throw
        self.convergence = single_convergence

        self.x = single_grating_center[0]
        self.y = single_grating_center[1]
        self.z = single_grating_center[2]


    def yield_grats(self, ind = None):
        if ind is not None:
            return [self.hub[ind],self.d[ind],self.gammy[ind],self.blaze[ind],
            self.yaw[ind],self.grat_length[ind],self.grat_width[ind],self.throw[ind],
            self.convergence[ind], self.x[ind], self.y[ind], self.z[ind]]
            
        else:
            return [self.hub,self.d,self.gammy,self.blaze,self.yaw,self.grat_length,
            self.grat_width, self.throw, self.convergence, self.x, self.y, self.z]
        

def trace_grating(ray_object, grating_object):

    init_rays = ray_object.yield_prays()
    orders = ray_object.order
    wavelengths = ray_object.wave
    weight = ray_object.weight

    hub = grating_object.hub
    gammy = grating_object.gammy
    yaw = grating_object.yaw
    d = grating_object.d

    conv_ang = grating_object.convergence * u.rad
    throw = grating_object.throw * u.mm


    glob_coords = [trans.tr.identity_matrix()] * 4 #just how coords works when using transform 
    #4 vector tracking the coordinate system 
    trans.transform(init_rays, 0, 0, 0, -conv_ang.to('rad').value, 0, 0, coords=glob_coords) #orthoganal to rays 
    
    # Go to hub location.
    trans.transform(init_rays, 0, 0, throw.to('mm').value, 0, 0, 0, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, -np.pi/2 + gammy.to('rad').value, 0, 0, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, 0, 0, np.pi, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, 0, 0, -yaw.to('rad').value, coords=glob_coords)
    
    #ind = np.where((rays[2] > (hub - grat_length/2).to('mm').value) & 
    #               (rays[2] < (hub + grat_length/2).to('mm').value))[0]
    
    surfaces.flat(init_rays)
    trans.transform(init_rays, 0, -hub.to('mm').value, 0, 0, 0, 0, coords=glob_coords)
    
    trans.reflect(init_rays)
    
    diff_inds = np.array([])
    
    #trans.radgrat(rays, d.to('nm').value/hub.to('mm').value, order, waves.to('nm').value)
    
    trans.radgrat(init_rays, d.to('nm').value/hub.to('mm').value, orders, wavelengths.to('nm').value)

    init_rays = trans.applyT(init_rays, glob_coords, inverse=True)
    #take inverse of glob coords 
    
    trans.transform(init_rays, 0, 0, 0, 0, 0, 0)
    surfaces.flat(init_rays)


    grating_ray_object = ray_object.yield_object_indices(ind = ones(len(init_rays[0]),dtype = bool))
    grating_ray_object.set_prays(init_rays)

    return grating_ray_object

    #return init_rays



def trace_grating_no_rowland(ray_object, grating_object, throw):

    init_rays = ray_object.yield_prays()
    orders = ray_object.order
    wavelengths = ray_object.wave
    weight = ray_object.weight

    hub = grating_object.hub
    gammy = grating_object.gammy
    yaw = grating_object.yaw
    d = grating_object.d

    conv_ang = grating_object.convergence * u.rad
    throw = throw * u.mm


    glob_coords = [trans.tr.identity_matrix()] * 4 #just how coords works when using transform 
    #4 vector tracking the coordinate system 
    trans.transform(init_rays, 0, 0, 0, -conv_ang.to('rad').value, 0, 0, coords=glob_coords) #orthoganal to rays 
    
    # Go to hub location.
    trans.transform(init_rays, 0, 0, throw.to('mm').value, 0, 0, 0, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, -np.pi/2 + gammy.to('rad').value, 0, 0, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, 0, 0, np.pi, coords=glob_coords)
    trans.transform(init_rays, 0, 0, 0, 0, 0, -yaw.to('rad').value, coords=glob_coords)
    
    #ind = np.where((rays[2] > (hub - grat_length/2).to('mm').value) & 
    #               (rays[2] < (hub + grat_length/2).to('mm').value))[0]
    
    surfaces.flat(init_rays)
    trans.transform(init_rays, 0, -hub.to('mm').value, 0, 0, 0, 0, coords=glob_coords)
    
    trans.reflect(init_rays)
    
    diff_inds = np.array([])
    
    #trans.radgrat(rays, d.to('nm').value/hub.to('mm').value, order, waves.to('nm').value)
    
    trans.radgrat(init_rays, d.to('nm').value/hub.to('mm').value, orders, wavelengths.to('nm').value)

    init_rays = trans.applyT(init_rays, glob_coords, inverse=True)
    #take inverse of glob coords 
    
    trans.transform(init_rays, 0, 0, 0, 0, 0, 0)
    surfaces.flat(init_rays)


    grating_ray_object = ray_object.yield_object_indices(ind = ones(len(init_rays[0]),dtype = bool))
    grating_ray_object.set_prays(init_rays)

    return grating_ray_object

    #return init_rays

