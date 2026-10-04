# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 08:19:18 2026

@author: Lewis
"""


#----------Imports----------#


import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os
from matplotlib import cm
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


cmap = cm.viridis.copy()


#----------Constants----------#


mu = np.pi*4e-7
a_coil = 9.5 * 1e-3 
a_wire = 1e-3 *(0.25/2)
deg_to_rad = np.pi/180
density_copper = 8950
density_ndfeb = 7500
MM=1e-3
g=9.81

#-----pathing---------#
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PRESETS_DIR = DATA_DIR / "Presets"
SENSOR_DATA_DIR = DATA_DIR / "Sensor Data"

#----------Global---------#

Is_Ideal = ""
Track_Used = ""
Track_Currents = ""
Pod_Used = ""
Pod_Angle = ""
Pod_Resolution = ""
Graph_Resolution = ""

#----------Maths Functions -----#
#Takes the roll pitch and yaw and produces a rotation matrix.
def rotation_matrix_from_euler(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Rotation about x (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    # Rotation about y (pitch)
    Ry = np.array([
        [ cp, 0, sp],
        [ 0,  1, 0],
        [-sp, 0, cp]
    ])

    # Rotation about z (yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])

    R = Rz @ Ry @ Rx
    return R


#----------Classes----------#


#The collection class is used to treat a set of magnets as one object
class Collection:
    def __init__(self,Magnets):
        self.Magnets = Magnets
        self.TotalMass = sum(magnet.mass for magnet in self.Magnets)
        self.TotalVolume = sum(magnet.volume for magnet in self.Magnets)
        self.Position = sum((magnet.position * magnet.mass) for magnet in self.Magnets) / self.TotalMass #Finds the centre of mass of all the magnets
        self.Pitch = 0
        self.Yaw = 0
        self.Roll = 0
        self.MagnetRings=[] #Contains list of lists of numbers identifying each magnet in Magnets with the same radii
        self.xRings = [] #Contains the radii of each of the lists in MagnetRings
        
        x_positions = []
        y_positions = []
        
        for Magnet in Magnets:
            x_positions.append(Magnet.position[0])
            y_positions.append(Magnet.position[1])
        x_max=max(x_positions)
        y_max=max(y_positions)
        
        #The following loop poppulates MagnetRings and xRings
        for i, Magnet in enumerate(Magnets):
            position = np.sqrt((Magnet.position[0]/x_max)**2+(Magnet.position[1]/y_max)**2)/np.sqrt(2)
            if position in self.xRings:
                for j, x in enumerate(self.xRings):
                    if position == np.sqrt((Magnet.position[0]/x_max)**2+(Magnet.position[1]/y_max)**2)/np.sqrt(2):
                        self.MagnetRings[j].append(i)
            else:
                self.xRings.append(position)
                self.MagnetRings.append([i])
                
                

    #This function changes the angle and position of each of the magnets so the object is rotated as one
    def ChangeAngle(self, roll, pitch, yaw):
        for Magnet in self.Magnets:
            R = rotation_matrix_from_euler(roll, pitch, yaw)
            Magnet.angle = R @ np.array((0,0,1))
            rel = Magnet.position - self.Position
            rel_rot = R @ rel
            Magnet.position = self.Position + rel_rot
    
    #This function calculates the inertia tensor for the object
    def compute_inertia_tensor(self):
        I_total = np.zeros((3, 3))
        COM = self.Position
        pitch = self.Pitch
        yaw = self.Yaw
        roll = self.Roll
        
        #Iterates through each magnet, calculating each magnets local inertia tensor
        for magnet in self.Magnets:
            mass = magnet.mass
            r = magnet.radius
            h = magnet.height
    
            #Magnet inertia about its own centre of mass
            Ixx = (1/12) * mass * (3*r**2 + h**2)
            Iyy = Ixx
            Izz = 0.5 * mass * r**2
    
            I_local = np.diag([Ixx, Iyy, Izz])
    
            R = rotation_matrix_from_euler(roll, pitch, yaw)
            I_world = R @ I_local @ R.T
    
            #Move the Inertia Tensor into the Obejct frame of reference
            d = magnet.position - COM
            d_outer = np.outer(d, d)
            I_shift = mass * ((np.dot(d, d) * np.eye(3)) - d_outer)
            I_total += I_world + I_shift
    
        return I_total

#The class which contains data on magnets, including height, radius, position, mass
class Magnet:
    def __init__(self, height, radius, position):
        self.height = float(height)
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=float)

        self.angle = np.array((0,0,1))

        self.volume = self.height * np.pi * self.radius**2
        self.mass = 0.0
        
    #Following function written by ChatGPT to check if any of the following points is contained within the magnet
    def contains(self, points):
        """
        Check whether points lie inside the cylindrical magnet,
        accounting for its orientation (self.angle).
        """
        points = np.atleast_2d(points)
    
        # Ensure axis is unit vector
        axis = np.asarray(self.angle, dtype=float)
        axis /= np.linalg.norm(axis)
    
        # Relative vectors from magnet centre
        rel = points - self.position
    
        # Axial projection (distance along cylinder axis)
        z_local = rel @ axis
    
        # Radial component (perpendicular to axis)
        radial_vec = rel - np.outer(z_local, axis)
        r = np.linalg.norm(radial_vec, axis=1)
    
        return (np.abs(z_local) <= self.height / 2) & (r <= self.radius)
        
#This is a child class of Magnet that takes in magnetic strength directly from data
class PermMagnet(Magnet):
    def __init__(self, height, radius, position, magnetic_strength):

        super().__init__(height, radius, position)

        #magnetic_strength is in MA/m
        self.magnetic_strength = float(magnetic_strength)/mu
        self.mass = density_ndfeb * self.volume

    #Computes the magnetic field at a set of points
    def field_at(self, points):
        points = np.atleast_2d(points)
        B = np.zeros((points.shape[0], 3))

        a = self.radius
        L = self.height
        b = L / 2
        zcentre = self.position[2]
        M = self.magnetic_strength

        for i, p in enumerate(points):
            x = p[0] - self.position[0]
            y = p[1] - self.position[1]
            z = p[2]

            Bx, By, Bz = B_calc(x=x, y=y, z=z,I=0.0,a=a,L=L,M=M,zcentre=zcentre,b=b) #Input magnet data and point and calculates the magnetic field

            B[i] = [Bx, By, Bz] #Adds that magnetic field to the larger array

        return B

#This is a child class of Magnet that takes in current and turns instead of magnetic strength
class ElectroMagnet(Magnet):
    def __init__(self, height, radius, position, current, turns,mu_r=3000, density_core=4800):
        super().__init__(height, radius, position)

        self.current = float(current)
        self.turns = int(turns)

        self.mu_r = float(mu_r)   # #relative permeability of core
        self.density_core = density_core

        #Approximate total mass = copper + core
        self.mass = density_copper * self.volume + density_core * self.volume

    #Computes the magnetic field at a set of points
    def field_at(self, points):
        points = np.atleast_2d(points)
        B = np.zeros((points.shape[0], 3))

        a = self.radius
        L = self.height
        b = L / 2
        zcentre = self.position[2]

        I = self.current
        NI = self.turns * I * self.mu_r

        for i, p in enumerate(points):
            x = p[0] - self.position[0]
            y = p[1] - self.position[1]
            z = p[2]

            Bx, By, Bz = B_calc(x=x, y=y, z=z,I=I,a=a,L=L,M=NI/L,zcentre=zcentre,b=b) #Input magnet data and point and calculates the magnetic field

            B[i] = [Bx, By, Bz] #Adds that magnetic field to the larger array

        return B


#----------Preset-Files----------#

#Loads data from the Track Presets file and lets user select which preset to use
def load_track_preset():
    df = pd.read_excel(PRESETS_DIR / "TrackPresets.xlsx")
    df = df.dropna(subset=['Setting Name'])
    presets = df.to_dict(orient='records')

    choice = int_choice("Select Track Preset:",[p['Setting Name'] for p in presets])
    
    preset = presets[choice]
    global Track_Used #Value used in file naming convention
    global Track_Currents  #Value used in file naming convention
    Track_Currents=""
    Track_Used = preset['Setting Name']
    row_list = []
    n_rows = int(preset['Rows'])
    
    #Iterates through each row in track and creates each magnet in the track
    for i in range(1, n_rows + 1):
        x = preset[f'Row {i} X'] * MM #Row X Value
        n_coils = int(preset[f'Row {i} Coil Number']) #Number of coils in row
        spacing = preset[f'Row {i} Coil Seperation'] * MM #Distance between coils
        radius = preset[f'Row {i} Coil Radius'] * MM #Radius of each magnet in row
        height = preset[f'Row {i} Coil Height'] * MM #Height of each magnet in row
        coil_type_str = preset[f'Row {i} Type'].lower() #Sets whether the magnet is permanent or an electromagnet
        
        if 'permanent' in coil_type_str:
            coil_type = 1
            magnetic_strength = preset[f'Row {i} Strength']
            Track_Currents = "Row " + str(i) + ": Default, "
        elif 'electro' in coil_type_str:
            coil_type = 2
            current = 0
            turns = int(preset[f'Row {i} Turns'])
            
        y_offsets = (np.arange(n_coils) - (n_coils - 1) / 2) * spacing  #Sets the y values of each of the coils
        row_coils = []
        for y in y_offsets:
            position = np.array([x, y, 0.0])
            if coil_type == 1:
                coil = PermMagnet(height, radius, position, magnetic_strength) #Creating the magnet as a permanent magnet
            else:
                coil = ElectroMagnet(height, radius, position, current, turns) #Creating the magnet as an electromagnet
            row_coils.append(coil)
        row_list.append({"row_index": i-1, "x_position": x, "coil_type": coil_type, "coils": row_coils}) #Makes a dictionary of each of the rows of magnets
    
    magnets = []
    for row in row_list:
        magnets.extend(row["coils"]) #Extends the dictionary into a single array
    return np.array(magnets)

#Loads data from the Pod Presets file and lets user select which preset to use
def load_pod_preset():
    df = pd.read_excel(PRESETS_DIR / "PodBasePresets.xlsx")
    df = df.dropna(subset=["Setting Name"])

    options = df["Setting Name"].tolist()
    choice = int_choice("Select pod preset:", options)

    row = df.iloc[choice]

    global Pod_Used
    Pod_Used = row["Setting Name"]

    magnets = []

    for i in range(1, 13):
        if pd.isna(row[f"X{i}"]):
            continue

        position = np.array([
            row[f"X{i}"] * MM,
            row[f"Y{i}"] * MM,
            row[f"Z{i}"] * MM,
        ])

        magnets.append(
            PermMagnet(
                height=row[f"Height {i}"] * MM,
                radius=row[f"Radius {i}"] * MM,
                position=position,
                magnetic_strength=-row[f"Strength {i}"],
            )
        )

    return magnets

#Loads data from the sensor data file
def load_sensor_data():
    sensor_locations = np.array([[0.0125, 0.025, 0.012],[-0.0125, 0.025, 0.1],[0.0125, 0.0, 0.012],[-0.0123, 0.0, 0.012],[-0.0125, 0.075, 0.012],[0.0125, 0.075, 0.012],[-0.0125, 0.05, 0.012],[0.0125, 0.05, 0.012]])
    
    df = pd.read_csv(SENSOR_DATA_DIR / "Train_mT2.csv")

    pod_positions = df[["podx", "pody", "podz"]].to_numpy()*1e-3

    sensor_fields = np.stack([df[[f"x{i}", f"y{i}", f"z{i}"]].to_numpy() for i in range(8)],axis=1)*1e-3

    return sensor_locations, pod_positions, sensor_fields

#This loads pre-generated potential energy data (dependent on a specific pod angle)
def load_potential_energy(path):
    data = pd.read_excel(path)
    
    #Takes the position data
    x_unique = np.sort(data['X_Pos'].unique())
    y_unique = np.sort(data['Y_Pos'].unique())
    z_unique = np.sort(data['Z_Pos'].unique())

    U_all = np.empty((len(x_unique), len(y_unique), len(z_unique)))
    U_all[:] = np.nan

    #Takes the index of each position
    x_idx = {x: i for i, x in enumerate(x_unique)}
    y_idx = {y: j for j, y in enumerate(y_unique)}
    z_idx = {z: k for k, z in enumerate(z_unique)}

    #Combined data into one array
    for _, row in data.iterrows():
        i = x_idx[row['X_Pos']]
        j = y_idx[row['Y_Pos']]
        k = z_idx[row['Z_Pos']]
        U_all[i, j, k] = row['U_Val']

    return U_all, x_unique, y_unique, z_unique

#This loads pre-generated magnetic field data
def load_magnetic_field(path):
    data = pd.read_excel(path)

    #Takes the position data
    x_unique = np.sort(data['X_Pos'].unique())
    y_unique = np.sort(data['Y_Pos'].unique())
    z_unique = np.sort(data['Z_Pos'].unique())

    shape = (len(x_unique), len(y_unique), len(z_unique))

    B_all = [
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan)
    ]

    #Takes the index of each position
    x_idx = {x: i for i, x in enumerate(x_unique)}
    y_idx = {y: j for j, y in enumerate(y_unique)}
    z_idx = {z: k for k, z in enumerate(z_unique)}
    
    #Combines data into one array
    for _, row in data.iterrows():
        i = x_idx[row['X_Pos']]
        j = y_idx[row['Y_Pos']]
        k = z_idx[row['Z_Pos']]

        B_all[0][i, j, k] = row['Bx']
        B_all[1][i, j, k] = row['By']
        B_all[2][i, j, k] = row['Bz']

    return B_all, x_unique, y_unique, z_unique


#----------Data-Files---------#

#Folder names
Plot_3D_TP_Folder = OUTPUT_DIR / "Plots" / "3D Plot Track & Pod"
Plot_2D_HM_Folder = OUTPUT_DIR / "Plots" / "2D Plot Heatmap"
Plot_2D_PT_Folder = OUTPUT_DIR / "Plots" / "2D Plot Position against Time"
Plot_3D_PT_Folder = OUTPUT_DIR / "Plots" / "3D Plot Position against Time"
Potential_Energy_Data_Folder = OUTPUT_DIR / "Saved_Magnetic_Field"
B_Field_Data_Folder = OUTPUT_DIR / "Saved_Magnetic_Field" / "Save B Field"
Ideal_Comparison_Folder = OUTPUT_DIR / "Ideal Comparison"
Sensor_Comparison_Folder = OUTPUT_DIR / "Sensor Comparison"
Saved_Currents_Folder = OUTPUT_DIR / "savedCurrents"

for folder in [
    Plot_3D_TP_Folder,
    Plot_2D_HM_Folder,
    Plot_2D_PT_Folder,
    Plot_3D_PT_Folder,
    Potential_Energy_Data_Folder,
    B_Field_Data_Folder,
    Ideal_Comparison_Folder,
    Sensor_Comparison_Folder,
    Saved_Currents_Folder,
]:
    Path(folder).mkdir(parents=True, exist_ok=True)

#Saves image to folder
def save_current_figure(folder):
    tag = "Track (" + Track_Used + ") Track Current (" + Track_Currents + ") Pod (" + Pod_Used + " " + Pod_Angle +")"
    filename = f"{tag}"
    path = os.path.join(folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")

#Saves potential energy data as an excel file
def save_potential_energy(U_all_in,x_data,y_data,z_data):
    tag = "Track (" + Track_Used+") Track Current ("+Track_Currents+") Pod ("+Pod_Used+" "+Pod_Angle+")"
    path = next((str(f) for f in Path(Potential_Energy_Data_Folder).glob("*.xls*") if f.stem == tag),"")
    if path !="":
        U_all_old, x_unique_old,y_unique_old,z_unique_old = load_potential_energy(path) #Loads old data if exists
    
    #Gets new position data
    x_unique_new = np.sort(np.unique(x_data))
    y_unique_new = np.sort(np.unique(y_data))
    z_unique_new = np.sort(np.unique(z_data))
    
    #Gets new field data
    U_all_new = np.empty((len(x_unique_new), len(y_unique_new), len(z_unique_new)))
    U_all_new[:] = np.nan
    
    x_idx = {x: i for i, x in enumerate(x_unique_new)}
    y_idx = {y: j for j, y in enumerate(y_unique_new)}
    z_idx = {z: k for k, z in enumerate(z_unique_new)}
    
    for i, x in enumerate(x_data):
        for j, y in enumerate(y_data):
            for k, z in enumerate(z_data):
                i_new = x_idx[x]
                j_new = y_idx[y]
                k_new = z_idx[z]
                U_all_new[i_new, j_new, k_new] = U_all_in[i, j, k]
   
    #If the old data exists, it is combined with the new data
    if path !="":
        U_all_old, x_unique_old,y_unique_old,z_unique_old = load_potential_energy(path)    
    
        # Union of coordinates
        x_combined = np.sort(np.unique(np.concatenate((x_unique_old, x_unique_new))))
        y_combined = np.sort(np.unique(np.concatenate((y_unique_old, y_unique_new))))
        z_combined = np.sort(np.unique(np.concatenate((z_unique_old, z_unique_new))))
    
        U_combined = np.full((len(x_combined),len(y_combined),len(z_combined)), np.nan)
    
        x_map = {x: i for i, x in enumerate(x_combined)}
        y_map = {y: j for j, y in enumerate(y_combined)}
        z_map = {z: k for k, z in enumerate(z_combined)}
    
        #Puts old data into the new map
        for i, x in enumerate(x_unique_old):
            for j, y in enumerate(y_unique_old):
                for k, z in enumerate(z_unique_old):
                    val = U_all_old[i, j, k]
                    if not np.isnan(val):
                        U_combined[x_map[x], y_map[y], z_map[z]] = val
    
        #Puts new data in, overwriting if it clashes with old data
        for i, x in enumerate(x_unique_new):
            for j, y in enumerate(y_unique_new):
                for k, z in enumerate(z_unique_new):
                    val = U_all_new[i, j, k]
                    if not np.isnan(val):
                        U_combined[x_map[x], y_map[y], z_map[z]] = val
    else:
        x_combined = x_unique_new
        y_combined = y_unique_new
        z_combined = z_unique_new
        U_combined = U_all_new

    X, Y, Z = np.meshgrid(x_combined,y_combined,z_combined,indexing='ij')

    df = pd.DataFrame({"X_Pos": X.ravel(),"Y_Pos": Y.ravel(),"Z_Pos": Z.ravel(),"U_Val": U_combined.ravel()})

    df = df.dropna(subset=["U_Val"])

    if path == "":
        filename = tag + ".xlsx"
    else:
        filename = path

    full_path = Path(Potential_Energy_Data_Folder) / filename

    df.to_excel(full_path, index=False)
    
#Save the comparison of potential energies as an excel file
def save_potential_energy_comparison(x_new_data, y_new_data, z_new_data, U_all_new, U_all_new_ideal, U_comparison):
    Is_Ideal = "[Comparison]"
    tag = Is_Ideal + " Track (" + Track_Used + ") Track Current (" + Track_Currents + ") Pod (" + Pod_Used + " " + Pod_Angle +")" #File Name
    folder = Path(Ideal_Comparison_Folder)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{tag}.xlsx"
    
    #Converts into 3d data
    X, Y, Z = np.meshgrid(x_new_data, y_new_data, z_new_data, indexing='ij')

    #Turns data into table form
    df_grid = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten(), 'U_val': U_all_new.flatten(), 'U_val_ideal': U_all_new_ideal.flatten(), 'U_comparison': U_comparison.flatten()})
    
    #Writes data to excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_grid.to_excel(writer, sheet_name='Grid_Data', index=False)
    print(f"Saved comparison → {file_path}")

#Saves the magnetic field data to an excel file
def save_magnetic_field(B_all_in, x_data, y_data, z_data):

    Path(B_Field_Data_Folder).mkdir(parents=True, exist_ok=True)

    tag = "Track (" + Track_Used + ") Track Current (" + Track_Currents + ") Pod (" + Pod_Used + " " + Pod_Angle + ")" #file name
    path = next((str(f) for f in Path(B_Field_Data_Folder).glob("*.xls*") if f.stem == tag), "")
    
    #The if statement checks if old data exists, and loads it if it does
    if path != "":
        B_old, x_unique_old, y_unique_old, z_unique_old = load_magnetic_field(path)

    #Takes new unique position data
    x_unique_new = np.sort(np.unique(x_data))
    y_unique_new = np.sort(np.unique(y_data))
    z_unique_new = np.sort(np.unique(z_data))

    shape_new = (len(x_unique_new), len(y_unique_new), len(z_unique_new))

    B_new = [np.full(shape_new, np.nan),np.full(shape_new, np.nan),np.full(shape_new, np.nan)]
    
    #Takes the index of each unique position data
    x_idx = {x: i for i, x in enumerate(x_unique_new)}
    y_idx = {y: j for j, y in enumerate(y_unique_new)}
    z_idx = {z: k for k, z in enumerate(z_unique_new)}

    #Creates new B_field data with correct indexing relating to position data
    for i, x in enumerate(x_data):
        for j, y in enumerate(y_data):
            for k, z in enumerate(z_data):
                i_new = x_idx[x]
                j_new = y_idx[y]
                k_new = z_idx[z]

                B_new[0][i_new, j_new, k_new] = B_all_in[0][i, j, k]
                B_new[1][i_new, j_new, k_new] = B_all_in[1][i, j, k]
                B_new[2][i_new, j_new, k_new] = B_all_in[2][i, j, k]

    #Combines new data with old if exists
    if path != "":

        x_combined = np.sort(np.unique(np.concatenate((x_unique_old, x_unique_new))))
        y_combined = np.sort(np.unique(np.concatenate((y_unique_old, y_unique_new))))
        z_combined = np.sort(np.unique(np.concatenate((z_unique_old, z_unique_new))))

        shape_combined = (len(x_combined), len(y_combined), len(z_combined))

        B_combined = [
            np.full(shape_combined, np.nan),
            np.full(shape_combined, np.nan),
            np.full(shape_combined, np.nan)
        ]

        x_map = {x: i for i, x in enumerate(x_combined)}
        y_map = {y: j for j, y in enumerate(y_combined)}
        z_map = {z: k for k, z in enumerate(z_combined)}

        #Puts old data into the new map
        for i, x in enumerate(x_unique_old):
            for j, y in enumerate(y_unique_old):
                for k, z in enumerate(z_unique_old):
                    for comp in range(3):
                        val = B_old[comp][i, j, k]
                        if not np.isnan(val):
                            B_combined[comp][x_map[x], y_map[y], z_map[z]] = val

        #Puts new data in, overwriting if it clashes with old data
        for i, x in enumerate(x_unique_new):
            for j, y in enumerate(y_unique_new):
                for k, z in enumerate(z_unique_new):
                    for comp in range(3):
                        val = B_new[comp][i, j, k]
                        if not np.isnan(val):
                            B_combined[comp][x_map[x], y_map[y], z_map[z]] = val

    else:
        x_combined = x_unique_new
        y_combined = y_unique_new
        z_combined = z_unique_new
        B_combined = B_new

    #Produce a flattened table
    X, Y, Z = np.meshgrid(x_combined, y_combined, z_combined, indexing='ij')

    df = pd.DataFrame({"X_Pos": X.ravel(),"Y_Pos": Y.ravel(),"Z_Pos": Z.ravel(),"Bx": B_combined[0].ravel(),"By": B_combined[1].ravel(),"Bz": B_combined[2].ravel(),})

    df = df.dropna(subset=["Bx", "By", "Bz"], how="all")

    filename = tag + ".xlsx" if path == "" else Path(path).name
    full_path = Path(B_Field_Data_Folder) / filename

    df.to_excel(full_path, index=False)

#Saves the currents used at different pod positions to an excel file
def savecurrents(Track, Pod_x, Pod_y, Pod_z, Magnet_Currents_xyz):
    
    save_dir = Saved_Currents_Folder
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / "magnet_currents.xlsx"

    #position of the magnets is put into a data frame
    magnet_ids = list(range(len(Track.Magnets)))
    magnet_x = [magnet.position[0] for magnet in Track.Magnets]
    magnet_y = [magnet.position[1] for magnet in Track.Magnets]

    magnet_df = pd.DataFrame({"Magnet_ID": magnet_ids,"x_position": magnet_x,"y_position": magnet_y})

    #current for each pod position is put into the data frame
    currents_df = pd.DataFrame(Magnet_Currents_xyz)

    currents_df.insert(0, "Pod_z", Pod_z)
    currents_df.insert(0, "Pod_y", Pod_y)
    currents_df.insert(0, "Pod_x", Pod_x)

    currents_df.columns = (["Pod_x", "Pod_y", "Pod_z"] +[f"Magnet_{i}_Current" for i in range(len(Track.Magnets))])

    with pd.ExcelWriter(file_path) as writer:
        magnet_df.to_excel(writer, sheet_name="Magnet_Positions", index=False)
        currents_df.to_excel(writer, sheet_name="Magnet_Currents", index=False)


#-------Simulations----#

#Calculates the magnetic field from both the track and pod
def Full_Field_At_Point(Pod,Track,point):
    B_field = np.zeros(3, dtype=float)
    for magnet in Pod.Magnets:
        B_field += magnet.field_at(point)[0]
    for magnet in Track.Magnets:
        B_field += magnet.field_at(point)[0]
    return B_field

#Calculates the magnetic field from just the track
def Field_On_Pod_Point(Track, point):
    B_field = np.zeros(3, dtype=float)
    for magnet in Track.Magnets:
        B_field += magnet.field_at(point)[0]
    return B_field

def rotate_point_into_pod_frame(sensor_pos, pod_pos, R):
    """
    Rotate a sensor position into the pod's frame to simulate pod tilt.
    """
    relative_pos = np.array(sensor_pos) - np.array(pod_pos)
    rotated_pos = R.T @ relative_pos  # Transpose = inverse for orthogonal R
    return rotated_pos

#Ideal field to cause restoring force
def B_field_Valley(Pod, point):
    scale = 5
    z_centre = 0.40
    x, y, z = point
        
    Bx = -x*(z-z_centre)
    By = -y*(z-z_centre)
    Bz = (x**2 + y**2 + (z-z_centre)**2)
    return scale*np.array([Bx, By, Bz])

#Ideal field to oppose gravity
def B_field_Gravity(Pod, point,point_num): 
    x, y, z = point
    M = Pod.TotalMass/point_num
    Bx = -0.5*M*g*(x)
    By = -0.5*M*g*(y)
    Bz = M*g*(z)
    return -np.array([Bx, By, Bz])

#Taken directly from Bella Mak's code
@numba.njit
def constants(rho,z,a,M,b):

    B_0 = (mu*M)/np.pi
    zplus = z + b
    zminus = z - b
    alphap = a/(np.sqrt(zplus**2 + (rho+a)**2))
    alpham = a/(np.sqrt(zminus**2 + (rho+a)**2))
    betap = zplus/(np.sqrt(zplus**2 + (rho+a)**2))
    betam = zminus/(np.sqrt(zminus**2 + (rho+a)**2))
    gamma = (a-rho)/(a+rho)
    gammasq = gamma**2
    kplus = np.sqrt((zplus**2 + (a-rho)**2)/(zplus**2 + (a+rho)**2))
    kminus = np.sqrt((zminus**2 + (a-rho)**2)/(zminus**2 + (a+rho)**2))

    return B_0,zplus,zminus,alphap,alpham,betap,betam,gamma,gammasq,kplus,kminus #rho

#Modified from Bella Mak's Code
def B_calc(x,y,z,I,a,L,M,zcentre,b):
    #Cylindrical coordinates (no rotation)
    rho = np.sqrt(x**2 + y**2)
    azimuth = np.arctan2(y, x)
    zref = z - zcentre
    
    #Calculate field in cylindrical coordinates
    B_0, zplus, zminus, alphap, alpham, betap, betam, gamma, gammasq, kplus, kminus = constants(rho, zref, a, M, b)
    
    Bz = B_z(B_0, a, rho, betap, kplus, gammasq, gamma, betam, kminus)
    Brho = B_rho(B_0, alphap, kplus, alpham, kminus)
    
    B_x = np.cos(azimuth) * Brho
    B_y = np.sin(azimuth) * Brho
    
    return (B_x, B_y, Bz)

#Taken Directly from Bella Mak's Code but with adjusted boundary clamping
@numba.njit
def elliptic(kc, p, c, s):
    if kc == 0:
        return np.nan
    errtol = .000001
    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.
    if p > 0:
        pp = np.sqrt(p)
        ss = s / pp
    else:
        f = kc * kc
        q = 1. - f
        g = 1. - pp
        f = f - pp
        q = q * (ss - c * pp)
        pp = np.sqrt(f / g)
        cc = (c - ss) / g
        ss = -q / (g * g * pp) + cc * pp
    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    while abs(g - k) > g * errtol:
        k = 2 * np.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em
        
    if kc < 1e-6:
        kc = 1e-6
    elif kc > 1.0 - 1e-10:
        print(kc)
        kc = 1.0 - 1e-10
    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))

#Taken Directly from Bella Mak's Code
@numba.njit
def B_rho(B0, alpha_p, k_plus, alpha_m, k_minus):
    # Regularised and symmetric radial contribution
    return B0 * (alpha_p * elliptic(k_plus, 1.0, 1.0, -1.0) -alpha_m * elliptic(k_minus, 1.0, 1.0, -1.0))

#Taken Directly from Bella Mak's Code but with adjusted boundary clamping
@numba.njit
def B_z(B0, a, rho, beta_p, k_plus, gamma_sq, gamma, beta_m, k_minus):
    # Regularised and more realistic toroidal field shape
    denominator = a + max(rho, 1e-15)
    return (B0 * a / denominator) * (beta_p * elliptic(k_plus, gamma_sq, 1.0, gamma) -beta_m * elliptic(k_minus, gamma_sq, 1.0, gamma))
    
#Modified from Bella Mak's code to calculate once for entire pod instead of one magnet at a time
def closed_cyl(x, y, z, Pod):
    Bx_total,By_total,Bz_total=0,0,0
    for magnet in Pod.Magnets:
        x0, y0, z0 = magnet.position
        Mz = magnet.magnetic_strength
    
        # cinverting into cylindrical coordinates
        dx = x - x0
        dy = y - y0
        rho = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) if rho > 1e-12 else 0.0
    
        # relative z coords of the top and bottom cylinder faces
        zt = z - (z0 + magnet.height/ 2)
        zb = z - (z0 - magnet.height / 2)
    
        a = magnet.radius
        gamma = (zt - zb) / 2
        gamma_sq = gamma ** 2
        B0 = mu * Mz / np.pi
    
        alpha_p = zt / np.sqrt((a + rho)**2 + zt**2)
        alpha_m = zb / np.sqrt((a + rho)**2 + zb**2)
    
        beta_p = (a**2 - rho**2 - zt**2) / ((a - rho)**2 + zt**2 + 1e-12)
        beta_m = (a**2 - rho**2 - zb**2) / ((a - rho)**2 + zb**2 + 1e-12)
    
        k_plus = np.sqrt(4 * a * rho / ((a + rho)**2 + zt**2))
        k_minus = np.sqrt(4 * a * rho / ((a + rho)**2 + zb**2))
    
        Bz_val = B_z(B0, a, rho, beta_p, k_plus, gamma_sq, gamma, beta_m, k_minus)
        Brho_val = B_rho(B0, alpha_p, k_plus, alpha_m, k_minus)
        
        if rho < 2e-6:
             Bz_val = np.clip(Bz_val, -1, 0.5)  # clip to 1 T if required            
        if rho < 1e-6:
             Brho_val = 0.0  
            
        Bx = Brho_val * np.cos(phi) if rho > 1e-12 else 0.0
        By = Brho_val * np.sin(phi) if rho > 1e-12 else 0.0
        Bx_total+=Bx
        By_total+=By
        Bz_total+=Bz_val

    return Bx_total, By_total, Bz_total


#----------Data_Comparison----------#

#Checks the sensor data against the simulated data
def Sensor_Check(Pod, Track):
    sensor_locations, pod_positions, sensor_fields = load_sensor_data()
    z_base = max(magnet.height for magnet in Track.Magnets)

    recorded_vectors = []
    simulated_vectors = []
    
    #Iterates through each position the pod takes
    for i, pod_position in tqdm(enumerate(pod_positions)):
        #Put the pod into the correct positions
        for magnet in Pod.Magnets:
            magnet.position += pod_position
            magnet.position[2] += z_base


        recorded_block = []
        simulated_block = []
        
        #Iterates through each sensor location
        for j, (x, y, z) in enumerate(sensor_locations):
            B_all = np.array([Field_On_Pod_Point(Track, (x, y, z)) + closed_cyl(x, y, z, Pod)]) #Calculates the magnetic field at each point
            ref_all = np.array([sensor_fields[i][j]])        
            simulated_block.append(B_all)
            recorded_block.append(ref_all)

        simulated_vectors.append(simulated_block)
        recorded_vectors.append(recorded_block)

        #Resets Pod into normal coordinates
        for magnet in Pod.Magnets:
            magnet.position -= pod_position
            magnet.position[2] -= z_base
    
    #Start ChatGPT
    def build_df(data):
        rows = []
        for block in data:
            row = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            for vec in block:
                row['x'] += vec[0]
                row['y'] += vec[1]
                row['z'] += vec[2]
            # Average over all points in the block
            row['x'] /= len(block)
            row['y'] /= len(block)
            row['z'] /= len(block)
            rows.append(row)
        return pd.DataFrame(rows)

    df_recorded = build_df(recorded_vectors)
    df_simulated = build_df(simulated_vectors)
    df_diff = df_simulated - df_recorded
    #End ChatGPT

    # Compute means and standard deviations of combined x, y, z
    summary_stats = {'Recorded_Mean': df_recorded.mean(),'Recorded_Std': df_recorded.std(),'Simulated_Mean': df_simulated.mean(),'Simulated_Std': df_simulated.std()}

    output_path = Sensor_Comparison_Folder / "sensor_sim_comparison.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_recorded.to_excel(writer, sheet_name='Recorded', index=False)
        df_simulated.to_excel(writer, sheet_name='Simulated', index=False)
        df_diff.to_excel(writer, sheet_name='Difference', index=False)
        pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Stats')

#Checks the Controlled system against the Ideal System
def Ideal_Check(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num):
    global Is_Ideal
    Is_Ideal ="[Controlled]"
    energy_potential_controlled(Pod, Track, pod_pos, pod_pts, x_min, x_max, x_num, y_min, y_max, y_num, z_min, z_max, z_num) #Gets the potential energy of the controlled Track
    tag = Is_Ideal+"  Track (" + Track_Used+") Track Current ("+Track_Currents+") Pod ("+Pod_Used+")"
    path = next((str(f) for f in Path(Potential_Energy_Data_Folder).glob("*.xls*") if f.stem == tag),"")
    U_all, x_data,y_data,z_data = load_potential_energy(path)
    Is_Ideal="[Ideal]"
    energy_potential_ideal(Pod, Track, pod_pos, pod_pts, x_min, x_max, x_num, y_min, y_max, y_num, z_min, z_max, z_num) #Gets the potential energy of the ideal Track
    tag = Is_Ideal+"  Track (" + Track_Used+") Track Current ("+Track_Currents+") Pod ("+Pod_Used+")"
    path = next((str(f) for f in Path(Potential_Energy_Data_Folder).glob("*.xls*") if f.stem == tag),"")
    U_all_ideal, x_data_ideal,y_data_ideal,z_data_ideal = load_potential_energy(path)
    
    i_track = []
    i_ideal = []
    
    #Iterates through all points in the x direction in the controlled field, checking if they match, within a certain tolerance, a similar point in the ideal field
    for i, x in enumerate(x_data):
        matches = np.where(np.isclose(x_data_ideal, x, rtol=1e-12))[0]
        
        if len(matches) > 0:
            i_track.append(i)
            i_ideal.append(matches[0])  # take the first match
    j_track = []
    j_ideal = []
    
    #Iterates through all points in the y direction in the controlled field, checking if they match, within a certain tolerance, a similar point in the ideal field
    for j, y in enumerate(y_data):
        matches = np.where(np.isclose(y_data_ideal, y, rtol=1e-12))[0]
        
        if len(matches) > 0:
            j_track.append(j)
            j_ideal.append(matches[0])  # take the first match
            
    k_track = []
    k_ideal = []
    
    #Iterates through all points in the z direction in the controlled field, checking if they match, within a certain tolerance, a similar point in the ideal field
    for k, z in enumerate(z_data):
        matches = np.where(np.isclose(z_data_ideal, z, rtol=1e-12))[0]
        
        if len(matches) > 0:
            k_track.append(k)
            k_ideal.append(matches[0])  # take the first match
    
    #Place to store the magnetic field data that exists at the same points in both the ideal and controlled simulation
    U_all_new =  np.zeros((len(i_track), len(j_track),len(k_track)))
    U_all_new_ideal = np.zeros((len(i_track), len(j_track),len(k_track)))
    U_comparison = np.zeros((len(i_track), len(j_track),len(k_track)))
    x_new_data = np.zeros(len(i_track))
    y_new_data = np.zeros(len(j_track))
    z_new_data = np.zeros(len(k_track))
    
    #Populating these new fields with the match x,y, and z coordinates
    for i1, i2 in enumerate(i_track):
        x_new_data[i1] = x_data[i2]
    for j1, j2 in enumerate(j_track):
        y_new_data[j1] = y_data[j2]
    for k1, k2 in enumerate(k_track):
        z_new_data[k1] = z_data[k2]
    
    #Populating the potential energy data into the relevant arrays
    for i1, i2 in enumerate(i_track):
        for j1, j2 in enumerate(j_track):
            for k1, k2 in enumerate (k_track):
                U_all_new[i1,j1,k1]=U_all[i2,j2,k2] #Stores the controlled system data with a matching point in ideal
                U_all_new_ideal[i1,j1,k1] = U_all_ideal[i_ideal[i1], j_ideal[j1], k_ideal[k1]] #Stores the ideal system data with a matching point in controlled
                a = U_all_new[i1, j1, k1]
                b = U_all_new_ideal[i1, j1, k1]
                if np.isfinite(a) and np.isfinite(b):
                    U_comparison[i1, j1, k1] = b - a
                else:
                    U_comparison[i1, j1, k1] = np.nan
    
    save_potential_energy_comparison(x_new_data,y_new_data,z_new_data,U_all_new,U_all_new_ideal,U_comparison)

#----------Potential Energy----------#

#Takes in the Pod data and splits it into approximately N points
def generate_pod_base_points(Pod, N=300):

    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []
    z_mins, z_maxs = [], []

    #Finds the limits of each magnet
    for magnet in Pod.Magnets:
        r = magnet.radius
        h = magnet.height
        cx, cy, cz = magnet.position

        x_mins.append(cx - r)
        x_maxs.append(cx + r)
        y_mins.append(cy - r)
        y_maxs.append(cy + r)
        z_mins.append(cz - h/2)
        z_maxs.append(cz + h/2)
        
    #Finds the limit of the pod
    x_min, x_max = min(x_mins), max(x_maxs)
    y_min, y_max = min(y_mins), max(y_maxs)
    z_min, z_max = min(z_mins), max(z_maxs)

    #Difference between maxima and minima
    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    #Estimates the density of the volume each point is referencing
    density = N / Pod.TotalVolume
    bbox_volume = Lx * Ly * Lz
    target_points = density * bbox_volume

    #determines how the pod should be spaced based on this density
    d = (bbox_volume / target_points) ** (1/3)

    #Number of spaces are determined by the length of the dimension being split
    nx = max(1, int(np.ceil(Lx / d)))
    ny = max(1, int(np.ceil(Ly / d)))
    nz = max(1, int(np.ceil(Lz / d)))

    #splits each dimension into nx,ny,nz spaces
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    zs = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    mask_total = np.zeros(len(grid), dtype=bool)
    
    #Goes through each magnet and checks that a point is within at least one of the magnets
    for magnet in Pod.Magnets:
        r = magnet.radius
        h = magnet.height
        cx, cy, cz = magnet.position

        dx = grid[:, 0] - cx
        dy = grid[:, 1] - cy
        dz = grid[:, 2] - cz

        radial = dx**2 + dy**2 <= r**2
        vertical = np.abs(dz) <= h/2

        mask_total |= (radial & vertical)
        
    #Returns only the correctly positioned points
    pod_pts = grid[mask_total]
    return pod_pts

#Calculates the magnetic field and potential energy of the ideal system
def energy_potential_ideal(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num):
    U_all_old = None
    x_unique_old = y_unique_old = z_unique_old = None
    
    global Is_Ideal
    Is_Ideal = "[Ideal]"
    tag = Is_Ideal+"  Track (" + Track_Used+") Pod ("+Pod_Used+")"
    path = next((str(f) for f in Path(Potential_Energy_Data_Folder).glob("*.xls*") if f.stem == tag),"")
    #Collects old energy potential data already calculated
    if path !="":
        U_all_old, x_unique_old,y_unique_old,z_unique_old = load_potential_energy(path)
    
    #Coordinate grids (centres)
    dx = (x_max-x_min)/x_num
    dy = (y_max-y_min)/y_num
    dz = (z_max-z_min)/z_num
    x_data = np.linspace(x_min+dx/2, x_max-dx/2, x_num)
    y_data = np.linspace(y_min+dy/2, y_max-dy/2, y_num)
    z_data = np.linspace(z_min+dz/2, z_max-dz/2, z_num)

    U_all = np.zeros((x_num, y_num,z_num))
    B_all = [np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num))]

    #Magnetic moment per dipole
    dV = Pod.TotalVolume / pod_pts.shape[0]
    M = Pod.Magnets[0].magnetic_strength
    m_vec = Pod.Magnets[0].angle * dV * M

    #TQDM progress bar
    total_steps = z_num * x_num * y_num
    pbar = tqdm(total=total_steps, desc="Computing potential energy", unit="pt")

    #Iterates through each position    
    for k, z in enumerate(z_data):
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                shift = np.array([x, y, z]) - pod_pos
                pts_shifted = pod_pts + shift
    
                #Check which points are inside any track magnet
                inside_any = np.zeros(len(pts_shifted), dtype=bool)
                for magnet in Track.Magnets:
                    inside_any |= magnet.contains(pts_shifted)
                #If there is older data that has already been calcualted, it is used instead
                if U_all_old is not None:
                    if (np.any(np.isclose(x_unique_old, x, rtol=1e-12)) and
                            np.any(np.isclose(y_unique_old, y, rtol=1e-12)) and
                            np.any(np.isclose(z_unique_old, z, rtol=1e-12))):
                        
                            i_old = np.where(np.isclose(x_unique_old, x, rtol=1e-12))[0][0]
                            j_old = np.where(np.isclose(y_unique_old, y, rtol=1e-12))[0][0]
                            k_old = np.where(np.isclose(z_unique_old, z, rtol=1e-12))[0][0]
                        
                            old_val = U_all_old[i_old, j_old, k_old]
                        
                            if not np.isnan(old_val):
                                U_all[i, j, k] = old_val
                    else:
                        if np.any(inside_any):
                            U_all[i, j,k] = np.nan
                        else:
                            U_all[ i, j,k] = Pod.TotalMass * 9.81 * (z)
                            for idx, pt in enumerate(pts_shifted):
                                #Magnetic field is calculated
                                Bx, By, Bz = B_field_Valley(Pod,pt)+B_field_Gravity(Pod, pt, len(pod_pts))
                                B_all[0][i,j,k] = Bx
                                B_all[1][i,j,k] = By
                                B_all[2][i,j,k] = Bz
                                U_all[i, j, k] += np.abs(np.dot(m_vec, np.array([B_all[0][i,j,k],B_all[1][i,j,k],B_all[2][i,j,k]])))
                else:
                    if np.any(inside_any):
                        U_all[i, j,k] = np.nan
                    else:
                        U_all[ i, j,k] = Pod.TotalMass * 9.81 * (z)
                        for idx, pt in enumerate(pts_shifted):
                                #Magnetic field is calculated
                                Bx, By, Bz = B_field_Valley(Pod,pt)+B_field_Gravity(Pod, pt, len(pod_pts))
                                B_all[0][i,j,k] = Bx
                                B_all[1][i,j,k] = By
                                B_all[2][i,j,k] = Bz
                                U_all[i, j, k] += np.abs(np.dot(m_vec, np.array([B_all[0][i,j,k],B_all[1][i,j,k],B_all[2][i,j,k]])))
                pbar.update(1)
    pbar.close()
    
    save_potential_energy(U_all, x_data, y_data, z_data)
    
    save_magnetic_field(B_all, x_data, y_data, z_data)
    
    plot_heatmap_with_gradient(U_all, x_data, y_data, z_data)

#Calculates the magnetic field and potential energy of the controlled system
def energy_potential_controlled(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num):
    global Is_Ideal
    Is_Ideal = "[Controlled]"
    #Coordinate grids (centres)
    dx = (x_max-x_min)/x_num
    dy = (y_max-y_min)/y_num
    dz = (z_max-z_min)/z_num
    x_data = np.linspace(x_min+dx/2, x_max-dx/2, x_num)
    y_data = np.linspace(y_min+dy/2, y_max-dy/2, y_num)
    z_data = np.linspace(z_min+dz/2, z_max-dz/2, z_num)

    U_all = np.zeros((x_num, y_num,z_num))
    B_all = [np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num))]

    #Magnetic moment per dipole
    dV = Pod.TotalVolume / pod_pts.shape[0]
    M = Pod.Magnets[0].magnetic_strength
    m_vec = Pod.Magnets[0].angle * dV * M

    #TQDM progress bar
    total_steps = z_num * x_num * y_num
    pbar = tqdm(total=total_steps, desc="Computing potential energy", unit="pt")
    
    #Factors used to adjust the controlled magnetic field
    Magnet_proportion = 78/len(Track.Magnets)
    Mass_proportion = Pod.TotalMass/0.1459
    z_proportion = 1
    
    magnets = Track.Magnets
    rings = Track.MagnetRings
    xRings = Track.xRings
    
    Magnet_Positions = []
    for Magnet in Track.Magnets:
        Magnet_Positions.append(Magnet.position)
    
    Pod_x = []
    Pod_y = []
    Pod_z = []
    Magnet_Currents_xyz = []
    
    #Iterates through each pod position
    for k, z in enumerate(z_data):
        grav_term =  Pod.TotalMass * 9.81 * (z)
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                x_y_pos = np.hypot(x, y)
                #Calculates the current of each electromagnet with the pod in this possition
                for ring_i, ring in enumerate(rings):
                    for magnet_i in ring:
                        magnets[magnet_i].current = (z_proportion * Mass_proportion * Magnet_proportion *(1.8 + 0.032 * (xRings[ring_i]*2.5 + 1221.08*x_y_pos)**2+ 39.34 * x_y_pos) / 1000)
                Pod_x.append(x)
                Pod_y.append(y)
                Pod_z.append(z)
                temp_currents = [m.current for m in magnets]
                Magnet_Currents_xyz.append(temp_currents) #Collects current data
                shift = np.array((x - pod_pos[0], y - pod_pos[1], z - pod_pos[2]))
                pts_shifted = pod_pts + shift
    
                #Check which points are inside any track magnet
                inside_any = np.zeros(len(pts_shifted), dtype=bool)
                inside_any = False
                for magnet in magnets:
                    if np.any(magnet.contains(pts_shifted)):
                        inside_any = True
                        break
                if np.any(inside_any):
                    U_all[i, j,k] = np.nan
                else:
                    U_all[ i, j,k] = grav_term
                    for idx, pt in enumerate(pts_shifted):
                        #Calculate the magnetic field at this point
                        Bx, By, Bz = Field_On_Pod_Point(Track,pt)
                        B_all[0][i,j,k] +=Bx
                        B_all[1][i,j,k] += By
                        B_all[2][i,j,k] += Bz
                        U_all[i, j, k] -= m_vec[0]*Bx + m_vec[1]*By + m_vec[2]*Bz #Convert magnetic field to potential energy
                pbar.update(1)
    pbar.close()
    
    save_potential_energy(U_all, x_data, y_data, z_data)
    
    save_magnetic_field(B_all, x_data, y_data, z_data)
    
    savecurrents(Track, Pod_x, Pod_y, Pod_z, Magnet_Currents_xyz)
    
    plot_heatmap_with_gradient(U_all, x_data, y_data, z_data)

#Calculates the magnetic field and potential energy of the static system
def energy_potential_static(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num):

    global Is_Ideal
    Is_Ideal = "[Static]"
    #Coordinate grids (centres)
    dx = (x_max-x_min)/x_num
    dy = (y_max-y_min)/y_num
    dz = (z_max-z_min)/z_num
    x_data = np.linspace(x_min+dx/2, x_max-dx/2, x_num)
    y_data = np.linspace(y_min+dy/2, y_max-dy/2, y_num)
    z_data = np.linspace(z_min+dz/2, z_max-dz/2, z_num)

    U_all = np.zeros((x_num, y_num,z_num))
    B_all = [np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num)),np.zeros((x_num, y_num,z_num))]

    # Magnetic moment per dipole
    dV = Pod.TotalVolume / pod_pts.shape[0]
    M = Pod.Magnets[0].magnetic_strength
    m_vec = Pod.Magnets[0].angle * dV * M
    

    #TQDM progress bar
    total_steps = z_num * x_num * y_num
    pbar = tqdm(total=total_steps, desc="Computing potential energy", unit="pt")

    #Factors used to adjust the controlled magnetic field
    Magnet_proportion = 78/len(Track.Magnets)
    Mass_proportion = Pod.TotalMass/0.1459
    z_proportion = 1
    
    magnets = Track.Magnets
    rings = Track.MagnetRings
    xRings = Track.xRings
    x_y_pos = 0

    Magnet_Positions = []
    for Magnet in Track.Magnets:
        Magnet_Positions.append(Magnet.position)
        
    for ring_i, ring in enumerate(rings):
        for magnet_i in ring:
            magnets[magnet_i].current = (z_proportion * Mass_proportion * Magnet_proportion *(1.8 + 0.032 * (xRings[ring_i]*2.5 + 1221.08*x_y_pos)**2+ 39.34 * x_y_pos) / 1000)
               
    
        
    #iterates through each pod position
    for k, z in enumerate(z_data):
        grav_term =  Pod.TotalMass * 9.81 * z
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                shift = np.array((x - pod_pos[0], y - pod_pos[1], z - pod_pos[2]))
                pts_shifted = pod_pts + shift
                #Check which points are inside any track magnet
                inside_any = np.zeros(len(pts_shifted), dtype=bool)
                inside_any = False
                for magnet in magnets:
                    if np.any(magnet.contains(pts_shifted)):
                        inside_any = True
                        break
                if np.any(inside_any):
                    U_all[i, j,k] = np.nan
                else:
                    U_all[ i, j,k] = grav_term
                    for idx, pt in enumerate(pts_shifted):
                        #Calculates the magnetic field at a point
                        Bx, By, Bz = Field_On_Pod_Point(Track,pt)
                        B_all[0][i,j,k] +=Bx
                        B_all[1][i,j,k] += By
                        B_all[2][i,j,k] += Bz
                        U_all[i, j, k] -= m_vec[0]*Bx + m_vec[1]*By + m_vec[2]*Bz #Calculates the potential energy
                pbar.update(1)
    pbar.close()
    
    save_potential_energy(U_all, x_data, y_data, z_data)
    
    save_magnetic_field(B_all, x_data, y_data, z_data)
    
    plot_heatmap_with_gradient(U_all, x_data, y_data, z_data)

#----------Plotting----------#

#Generates image of track and pod setup
def plot_track_and_pod_3D(Track,Pod):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    #ChatGPT start
    def draw_cylinder(ax, centre, radius, height, angle, color='gray', alpha=1.0, resolution=12):
    
        x0, y0, z0 = centre
    
        # Normalize direction vector
        direction = np.array(angle, dtype=float)
        direction /= np.linalg.norm(direction)
    
        # Create cylinder aligned with Z-axis
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(0, height, resolution)
        Theta, Z = np.meshgrid(theta, z)
    
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)
        Z = Z
    
        # Stack coordinates
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    
        # Default axis
        z_axis = np.array([0, 0, 1])
    
        # If already aligned, skip rotation
        if not np.allclose(direction, z_axis):
            v = np.cross(z_axis, direction)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, direction)
    
            # Skew-symmetric cross-product matrix
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
    
            # Rodrigues' rotation formula
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    
            points = R @ points
    
        # Reshape rotated coordinates
        X = points[0].reshape((resolution, resolution)) + x0
        Y = points[1].reshape((resolution, resolution)) + y0
        Z = points[2].reshape((resolution, resolution)) + z0
    
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor='none')
    #Chatgpt End
    
    #Draws a cylinder for each track magnet
    for magnet in Track.Magnets:
        colour = ""
        if isinstance(magnet,PermMagnet): 
            colour = "skyblue"
        else:
            colour = "yellow"
        draw_cylinder(ax,magnet.position,magnet.radius,magnet.height,magnet.angle, color = colour, alpha=0.8)

    #Draws a cylinder for each pod magnet
    for magnet in Pod.Magnets:
        colour = ""
        if isinstance(magnet,PermMagnet): 
            colour = "skyblue"
        else:
            colour = "yellow"
        draw_cylinder(ax,magnet.position,magnet.radius,magnet.height,magnet.angle, color = colour, alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')        
    ax.set_title(f"Pod position: x={Pod.Position[0]:.3f}, y={Pod.Position[1]:.3f}, z={Pod.Position[2]:.3f} ")
    ax.legend()
    ax.set_box_aspect([1, 2, 0.5])  # Adjust for track proportions
    ax.view_init(elev=20, azim=120)
    ax.grid(True)
    plt.tight_layout()
    save_current_figure(Plot_3D_TP_Folder)
    plt.show()

#Generates the heatmap of the potential energy
def plot_heatmap_with_gradient(U_all, x_data, y_data, z_data,arrow_stride=2,arrow_scale=None):
    
    #Sets the colour for the heatmap
    vmin, vmax = 0, 1
    value_colours = [(0, "darkblue"),(0.1, "lightblue"),(0.2, "yellow"),(0.4, "orange"),(0.6, "red"),(0.8, "purple"),(1, "black"),]
    colours = [((v - vmin) / (vmax - vmin), c)for v, c in value_colours if vmin <= v <= vmax]
    cmap = LinearSegmentedColormap.from_list("fixed_map", colours)
    norm = Normalize(vmin=vmin, vmax=vmax)
    x_num, y_num, z_num = len(x_data), len(y_data), len(z_data)
    
    #ChatGPT fix for colour map error
    def safe_extent(vmin, vmax):
        if vmin == vmax:
            eps = abs(vmin)*1e-6 + 1e-9
            return vmin - eps, vmax + eps
        return vmin, vmax

    x_min, x_max = safe_extent(min(x_data), max(x_data))
    y_min, y_max = safe_extent(min(y_data), max(y_data))
    z_min, z_max = safe_extent(min(z_data), max(z_data))
    #ChatGPT end
    
    #These will be gradient values
    dU_dx = np.zeros_like(U_all)
    dU_dy = np.zeros_like(U_all)
    dU_dz = np.zeros_like(U_all)
    
    axes = []
    spacing = []
    
    #Only maps the gradient if there is multiple data points in that axis (otherwise gradient couldnt exist)
    if x_num > 1:
        axes.append(0)
        spacing.append(x_data)
    if y_num > 1:
        axes.append(1)
        spacing.append(y_data)
    if z_num > 1:
        axes.append(2)
        spacing.append(z_data)

    
    #Calculates the gradients for points
    if len(axes)>0:
        grads = np.gradient(U_all, *spacing, axis=axes, edge_order=2)
        g_index = 0
        if x_num > 1:
            dU_dx = grads[g_index]
            g_index += 1
        if y_num > 1:
            dU_dy = grads[g_index]
            g_index += 1
        if z_num > 1:
            dU_dz = grads[g_index]


    #Compute magnitude of gradient
    mag = np.sqrt(dU_dx**2 + dU_dy**2 + dU_dz**2)
    
    mag[mag == 0] = 1
            
    
    #Normalize so all arrows are same length
    dU_dx = dU_dx / mag
    dU_dy = dU_dy / mag
    dU_dz = dU_dz / mag

    cols = max(x_num, y_num, z_num)
    fig = plt.figure(figsize=(4 * cols, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, cols)

    #Plots the x-y plane with a fixed z value
    for k in range(z_num):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(U_all[:, :, k].T,origin="lower",extent=[x_min, x_max, y_min, y_max],cmap=cmap,norm=norm,interpolation="nearest",aspect="auto")

        if x_num > 1 and y_num > 1:
            X, Y = np.meshgrid(x_data, y_data, indexing="ij")
            #plots an arrow at at every other position
            ax.quiver(X[::arrow_stride, ::arrow_stride],Y[::arrow_stride, ::arrow_stride],-dU_dx[:, :, k][::arrow_stride, ::arrow_stride],-dU_dy[:, :, k][::arrow_stride, ::arrow_stride],color="white", scale=arrow_scale)

        ax.set_title(f"z = {z_data[k]*1e3:.1f} mm")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    #Plots the x-z plane with a fixed y value
    for j in range(y_num):
        ax = fig.add_subplot(gs[1, j])

        ax.imshow(U_all[:, j, :].T,
                  origin="lower",
                  extent=[x_min, x_max, z_min, z_max],
                  cmap=cmap,
                  norm=norm,
                  interpolation="nearest",
                  aspect="auto")

        if x_num > 1 and z_num > 1:
            X, Z = np.meshgrid(x_data, z_data, indexing="ij")
            #plots an arrow at at every other position
            ax.quiver( X[::arrow_stride, ::arrow_stride],Z[::arrow_stride, ::arrow_stride],-dU_dx[:, j, :][::arrow_stride, ::arrow_stride],-dU_dz[:, j, :][::arrow_stride, ::arrow_stride],color="white",scale=arrow_scale)
        ax.set_title(f"y = {y_data[j]*1e3:.1f} mm")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    #Plots the y-z plane with a fixed x value
    for i in range(x_num):
        ax = fig.add_subplot(gs[2, i])

        ax.imshow(U_all[i, :, :].T,origin="lower",extent=[y_min, y_max, z_min, z_max],cmap=cmap,norm=norm,interpolation="nearest",aspect="auto")

        if y_num > 1 and z_num > 1:
            Y, Z = np.meshgrid(y_data, z_data, indexing="ij")
            #plots an arrow at at every other position
            ax.quiver(Y[::arrow_stride, ::arrow_stride],Z[::arrow_stride, ::arrow_stride],-dU_dy[i, :, :][::arrow_stride, ::arrow_stride],-dU_dz[i, :, :][::arrow_stride, ::arrow_stride],color="white",scale=arrow_scale)

        ax.set_title(f"x = {x_data[i]*1e3:.1f} mm")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")

    #Produces the colour bar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable,ax=fig.axes,shrink=0.95,label="Potential Energy (J)")

    fig.suptitle("Potential Energy with Gradient Field")

    save_current_figure(Plot_2D_HM_Folder)
    plt.show()

#Calculates the pod position against time and produces graphs of it
def plot_movement(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num):
    global Is_Ideal
    Is_Ideal = "[Controlled]"
    tag = Is_Ideal + " Track (" + Track_Used + ") Pod (" + Pod_Used + " " + Pod_Angle +")"
    
    path = next((str(f) for f in Path(B_Field_Data_Folder).glob("*.xls*") if f.stem == tag), "")
    if not path:
        energy_potential_controlled(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num)
        # search again after generation
        path = next((str(f) for f in Path(B_Field_Data_Folder).glob("*.xls*") if f.stem == tag), "")

    B_all, x_data, y_data, z_data = load_magnetic_field(path)
    Bx = B_all[0]
    By = B_all[1]
    Bz = B_all[2]
    
    # Magnetic moment per dipole
    dV = Pod.TotalVolume / pod_pts.shape[0]
    M = Pod.Magnets[0].magnetic_strength
    m_vec = Pod.Magnets[0].angle * dV * M
    
    

    dx = (x_max - x_min)/(x_num-1) if x_num > 1 else 1.0
    dy = (y_max - y_min)/(y_num-1) if y_num > 1 else 1.0
    dz = (z_max - z_min)/(z_num-1) if z_num > 1 else 1.0
    
    #ChatGPT fix for calculating gradient when no gradient could exist
    def safe_gradient(F, dx, dy, dz):
        shape = F.shape

        # X-gradient
        if shape[0] > 1:
            dF_dx = np.gradient(F, dx, axis=0)
        else:
            dF_dx = np.zeros_like(F)
    
        # Y-gradient
        if shape[1] > 1:
            dF_dy = np.gradient(F, dy, axis=1)
        else:
            dF_dy = np.zeros_like(F)
    
        # Z-gradient
        if shape[2] > 1:
            dF_dz = np.gradient(F, dz, axis=2)
        else:
            dF_dz = np.zeros_like(F)
    
        return dF_dx, dF_dy, dF_dz
    #End of chatgpt
    
    dBx_dx, dBx_dy, dBx_dz = safe_gradient(Bx, dx, dy, dz)
    dBy_dx, dBy_dy, dBy_dz = safe_gradient(By, dx, dy, dz)
    dBz_dx, dBz_dy, dBz_dz = safe_gradient(Bz, dx, dy, dz)

    #The following interpolates all values for the magnetic fields and the gradients in each direction
    Bx_i = RegularGridInterpolator((x_data,y_data,z_data), Bx, bounds_error=False, fill_value=0.0)
    By_i = RegularGridInterpolator((x_data,y_data,z_data), By, bounds_error=False, fill_value=0.0)
    Bz_i = RegularGridInterpolator((x_data,y_data,z_data), Bz, bounds_error=False, fill_value=0.0)

    dBx_dx_i = RegularGridInterpolator((x_data,y_data,z_data), dBx_dx, bounds_error=False, fill_value=0.0)
    dBx_dy_i = RegularGridInterpolator((x_data,y_data,z_data), dBx_dy, bounds_error=False, fill_value=0.0)
    dBx_dz_i = RegularGridInterpolator((x_data,y_data,z_data), dBx_dz, bounds_error=False, fill_value=0.0)

    dBy_dx_i = RegularGridInterpolator((x_data,y_data,z_data), dBy_dx, bounds_error=False, fill_value=0.0)
    dBy_dy_i = RegularGridInterpolator((x_data,y_data,z_data), dBy_dy, bounds_error=False, fill_value=0.0)
    dBy_dz_i = RegularGridInterpolator((x_data,y_data,z_data), dBy_dz, bounds_error=False, fill_value=0.0)

    dBz_dx_i = RegularGridInterpolator((x_data,y_data,z_data), dBz_dx, bounds_error=False, fill_value=0.0)
    dBz_dy_i = RegularGridInterpolator((x_data,y_data,z_data), dBz_dy, bounds_error=False, fill_value=0.0)
    dBz_dz_i = RegularGridInterpolator((x_data,y_data,z_data), dBz_dz, bounds_error=False, fill_value=0.0)

    print("Enter starting position")
    x0 = float(input("x0: "))
    y0 = float(input("y0: "))
    z0 = float(input("z0: "))

    vx0 = vy0 = vz0 = float(0)
    roll0 = pitch0 = yaw0 = float(0)
    wx0 = wy0 = wz0 = float(0)

    m = Pod.TotalMass
    I = Pod.compute_inertia_tensor()
    I_inv = np.linalg.inv(I)
    m_body = m_vec
    
    #this function determines if the pod would be outside the simulated range
    def out_of_bounds(t, s):
        x, y, z = s[0], s[1], s[2]
        return min(x - x_min,x_max - x,y - y_min,y_max - y,z - z_min,z_max - z)
    
    out_of_bounds.terminal = True
    out_of_bounds.direction = -1
    
    #Function that takes in the time and the state of the system to perform a time step
    def dynamics(t, state):
        x,y,z,vx,vy,vz, roll,pitch,yaw, wx,wy,wz = state

        B = np.array([Bx_i((x,y,z)),By_i((x,y,z)),Bz_i((x,y,z))]).flatten()
        R = rotation_matrix_from_euler(roll,pitch,yaw)
        m_world = R @ m_body

        #Force is calculated as the dot product of magnetic moment with the gradient of the magnetic field
        Fx = (m_world[0]*dBx_dx_i((x,y,z)) +m_world[1]*dBy_dx_i((x,y,z)) +m_world[2]*dBz_dx_i((x,y,z)))
        Fy = (m_world[0]*dBx_dy_i((x,y,z)) +m_world[1]*dBy_dy_i((x,y,z)) + m_world[2]*dBz_dy_i((x,y,z)))
        Fz = (m_world[0]*dBx_dz_i((x,y,z)) +m_world[1]*dBy_dz_i((x,y,z)) +m_world[2]*dBz_dz_i((x,y,z)))-m*g
        F = np.array([Fx, Fy, Fz]).flatten()
        ax, ay, az = F / m

        #Torque is the cross product of the magnetic moment and the magnetic field at that point
        tau = np.cross(m_world, B)
        alpha = I_inv @ tau

        return [vx, vy, vz, ax, ay, az,wx, wy, wz,alpha[0], alpha[1], alpha[2]]
    
    #This is the starting state
    state_start = [x0,y0,z0,vx0,vy0,vz0,roll0,pitch0,yaw0,wx0,wy0,wz0]

    sol = solve_ivp(dynamics, [0, 2.0], state_start, max_step=0.001, events=out_of_bounds)
    
    #output all of the solutions into their arrays
    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    
    roll  = sol.y[6]
    pitch = sol.y[7]
    yaw   = sol.y[8]
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    
    #Plots of position data against time
    axs[0,0].plot(t, x)
    axs[0,0].set_ylabel("x (m)")
    
    axs[1,0].plot(t, y)
    axs[1,0].set_ylabel("y (m)")
    
    axs[2,0].plot(t, z)
    axs[2,0].set_ylabel("z (m)")
    axs[2,0].set_xlabel("Time (s)")
    
    #Plots of angle against time
    axs[0,1].plot(t, roll)
    axs[0,1].set_ylabel("roll (rad)")
    
    axs[1,1].plot(t, pitch)
    axs[1,1].set_ylabel("pitch (rad)")
    
    axs[2,1].plot(t, yaw)
    axs[2,1].set_ylabel("yaw (rad)")
    axs[2,1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    
    save_current_figure(Plot_2D_PT_Folder)

    plt.show()


#----------Interface----------#


#Function that lets the user select what data they are testing. Not all is used in each test
def potential_choice():
    #Selecting the pod
    Pod = Collection(load_pod_preset())
    
    #Setting the pod angle
    roll = float(input("Roll: "))
    pitch = float(input("Pitch: "))
    yaw = float(input("Yaw: "))
    Pod.ChangeAngle(roll, pitch, yaw)
    
    global Pod_Angle
    Pod_Angle=("[Theta "+str(roll)+", "+str(pitch)+", "+str(yaw)+"]").replace(".","_")
        
    #Selecting the track
    Track = Collection(load_track_preset())
    pod_pos=Pod.Position
    pod_pts = generate_pod_base_points(Pod)
    
    #Selecting what region of the system to model and with what resolution
    x_max = 0.5*float(input("Range in X values: "))
    x_min = -x_max
    x_num = int(input("Number of points in X axis: "))
    
    y_max = 0.5*float(input("Range in Y values: "))
    y_min = -y_max
    y_num = int(input("Number of points in Y axis: "))
    
    z_track_top = max(m.position[2] + m.height/2 for m in Track.Magnets)
    gap_between_track_top_pod_COM = Pod.Position[2]-z_track_top
    print()
    print("The Centre of Mass (COM) of the Pod is ", str(Pod.Position[2]))
    print("The Top of the Track is ", str(gap_between_track_top_pod_COM), " below the pod COM")
    print()
    z_max = 0.5*float(input("Range in Z values about Pod COM: "))
    z_min = -z_max
    z_max += Pod.Position[2]
    z_min += Pod.Position[2]
    z_num = int(input("Number of points in Z axis: "))
    
    return Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num

#Warning for certain bad inputs
def bad_input(issue):
    print()
    print("#----------Error----------#")
    print()
    print(issue)
    print()
    print("#----------Interface----------#")
    print()

#Function to insure correct integer is chosen
def int_choice(firstphrase, choices):
    choice = -1
    while choice == -1:
        print(firstphrase)
        for i, choice in enumerate(choices):
            print((i+1), " - ", choice)
        try:
            choice = int(input())-1
            if choice<0 or choice>=len(choices):
                bad_input("Input out of range.")
                choice = -1
            else:
                return choice
        except:
            bad_input("Input must be an integer.")
            choice = -1

#The interface for which test is performed
def Interface():
    quit = False
    while quit == False:
        choice = int_choice("Select Action:", ["Plot Track and Pod in 3D","Check previous model against recorded values","Plot energy potential", "Plot pod movement","Compare energy potential to theory","Quit"])
        if choice == 0:
            Track = Collection(load_track_preset())
            Pod = Collection(load_pod_preset())
            roll = float(input("Roll: "))
            pitch = float(input("Pitch: "))
            yaw = float(input("Yaw: "))
            
            global Pod_Angle
            Pod_Angle=("[Theta "+str(roll)+", "+str(pitch)+", "+str(yaw)+"]").replace(".","_")
            
            Pod.ChangeAngle(roll, pitch, yaw)
            plot_track_and_pod_3D(Track, Pod)
        elif choice == 1:
            Sensor_Check(Collection(load_pod_preset()), Collection(load_track_preset()))
        elif choice == 2:
            choice2 = int_choice("Select Option:",["Model Track Preset Static","Model Track Prest Controlled","Model Ideal Field"])
            if choice2==0:
                Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num = potential_choice()
                energy_potential_static(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num)
            elif choice2 == 1:
                Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num = potential_choice()
                energy_potential_controlled(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num)
            else:
                Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num = potential_choice()
                energy_potential_ideal(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num)
        elif choice == 3:
            Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num = potential_choice()
            plot_movement(Pod, Track, pod_pos, pod_pts, x_min, x_max, x_num, y_min, y_max, y_num, z_min, z_max, z_num)
        elif choice == 4:
            Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num = potential_choice()
            Ideal_Check(Pod, Track, pod_pos, pod_pts,x_min, x_max, x_num,y_min, y_max, y_num,z_min, z_max, z_num)
        else:
            quit = True




if __name__ == "__main__":
    Interface()