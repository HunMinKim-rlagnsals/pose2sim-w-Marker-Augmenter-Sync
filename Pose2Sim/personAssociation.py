#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## TRACKING OF PERSON OF INTEREST                                        ##
###########################################################################

Openpose detects all people in the field of view. 
- multi_person = false: Triangulates the most prominent person
- multi_person = true: Triangulates persons across views
                       Tracking them across time frames is done in the triangulation stage.

If multi_person = false, this module tries all possible triangulations of a chosen
anatomical point, and chooses the person for whom the reprojection error is smallest. 

If multi_person = true, it computes the distance between epipolar lines (camera to 
keypoint lines) for all persons detected in all views, and selects the best correspondences. 
The computation of the affinity matrix from the distance is inspired from the EasyMocap approach.

INPUTS: 
- a calibration file (.toml extension)
- json files from each camera folders with several detected persons
- a Config.toml file
- a skeleton model

OUTPUTS: 
- json files for each camera with only one person of interest
'''


## INIT
import os
import glob
import fnmatch
import re
import numpy as np
import json
import itertools as it
import toml
from tqdm import tqdm
import cv2
from anytree import RenderTree
from anytree.importer import DictImporter
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, sort_stringlist_by_last_number, natural_sort_key
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def persons_combinations(json_files_framef):
    '''
    Find all possible combinations of detected persons' ids. 
    Person's id when no person detected is set to -1.
    
    INPUT:
    - json_files_framef: list of strings

    OUTPUT:
    - personsIDs_comb: array, list of lists of int
    '''
    
    n_cams = len(json_files_framef)
    
    # amount of persons detected for each cam
    nb_persons_per_cam = []
    for c in range(n_cams):
        try:
            with open(json_files_framef[c], 'r') as js:
                nb_persons_per_cam += [len(json.load(js)['people'])]
        except:
            nb_persons_per_cam += [0]
    
    # persons combinations
    id_no_detect = [i for i, x in enumerate(nb_persons_per_cam) if x == 0]  # ids of cameras that have not detected any person
    nb_persons_per_cam = [x if x != 0 else 1 for x in nb_persons_per_cam] # temporarily replace persons count by 1 when no detection
    range_persons_per_cam = [range(nb_persons_per_cam[c]) for c in range(n_cams)] 
    personsIDs_comb = np.array(list(it.product(*range_persons_per_cam)), float) # all possible combinations of persons' ids
    personsIDs_comb[:,id_no_detect] = np.nan # -1 = persons' ids when no person detected
    
    return personsIDs_comb


def triangulate_comb(comb, coords, P_all, calib_params, config_dict):
    '''
    Triangulate 2D points and compute reprojection error for a combination of cameras.
    INPUTS:
    - comb: list of ints: combination of persons' ids for each camera
    - coords: array: x, y, likelihood for each camera
    - P_all: list of arrays: projection matrices for each camera
    - calib_params: dict: calibration parameters
    - config_dict: dictionary from Config.toml file
    OUTPUTS:
    - error_comb: float: reprojection error
    - comb: list of ints: combination of persons' ids for each camera
    - Q_comb: array: 3D coordinates of the triangulated point
    ''' 

    undistort_points = config_dict.get('triangulation').get('undistort_points')
    likelihood_threshold = config_dict.get('personAssociation').get('likelihood_threshold_association')

    # Replace likelihood by 0. if under likelihood_threshold
    coords[:,2][coords[:,2] < likelihood_threshold] = 0.
    comb[coords[:,2] == 0.] = np.nan

    # Filter coords and projection_matrices containing nans
    coords_filt = [coords[i] for i in range(len(comb)) if not np.isnan(comb[i])]
    projection_matrices_filt = [P_all[i] for i in range(len(comb)) if not np.isnan(comb[i])]
    if undistort_points:
        calib_params_R_filt = [calib_params['R'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_T_filt = [calib_params['T'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_K_filt = [calib_params['K'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_dist_filt = [calib_params['dist'][i] for i in range(len(comb)) if not np.isnan(comb[i])]

    # Triangulate 2D points
    try:
        x_files_filt, y_files_filt, likelihood_files_filt = np.array(coords_filt).T
        Q_comb = weighted_triangulation(projection_matrices_filt, x_files_filt, y_files_filt, likelihood_files_filt)
    except:
        Q_comb = [np.nan, np.nan, np.nan, 1.]

    # Reprojection
    if undistort_points:
        coords_2D_kpt_calc_filt = [cv2.projectPoints(np.array(Q_comb[:-1]), calib_params_R_filt[i], calib_params_T_filt[i], calib_params_K_filt[i], calib_params_dist_filt[i])[0] for i in range(len(Q_comb))]
        x_calc = [coords_2D_kpt_calc_filt[i][0,0,0] for i in range(len(Q_comb))]
        y_calc = [coords_2D_kpt_calc_filt[i][0,0,1] for i in range(len(Q_comb))]
    else:
        x_calc, y_calc = reprojection(projection_matrices_filt, Q_comb)

    # Reprojection error
    error_comb_per_cam = []
    for cam in range(len(x_calc)):
        q_file = (x_files_filt[cam], y_files_filt[cam])
        q_calc = (x_calc[cam], y_calc[cam])
        error_comb_per_cam.append( euclidean_distance(q_file, q_calc) )
    error_comb = np.mean(error_comb_per_cam)

    return error_comb, comb, Q_comb


def best_persons_and_cameras_combination(config_dict, json_files_framef, personsIDs_combinations, projection_matrices, tracked_keypoint_id, calib_params):
    '''
    Chooses the right person among the multiple ones found by
    OpenPose & excludes cameras with wrong 2d-pose estimation.
    
    1. triangulate the tracked keypoint for all possible combinations of people,
    2. compute difference between reprojection & original openpose detection,
    3. take combination with smallest error OR all those below the error threshold
    If error is too big, take off one or several of the cameras until err is 
    lower than "max_err_px".
    
    INPUTS:
    - a Config.toml file
    - json_files_framef: list of strings
    - personsIDs_combinations: array, list of lists of int
    - projection_matrices: list of arrays
    - tracked_keypoint_id: int

    OUTPUTS:
    - errors_below_thresh: list of float
    - comb_errors_below_thresh: list of arrays of ints
    '''
    
    error_threshold_tracking = config_dict.get('personAssociation').get('single_person').get('reproj_error_threshold_association')
    min_cameras_for_triangulation = config_dict.get('triangulation').get('min_cameras_for_triangulation')
    undistort_points = config_dict.get('triangulation').get('undistort_points')

    n_cams = len(json_files_framef)
    error_min = np.inf 
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    Q_kpt = []
    while error_min > error_threshold_tracking and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Try all persons combinations
        for combination in personsIDs_combinations:
            #  Get coords from files
            coords = []
            for index_cam, person_nb in enumerate(combination):
                try:
                    js = read_json(json_files_framef[index_cam])
                    coords.append(js[int(person_nb)][tracked_keypoint_id*3:tracked_keypoint_id*3+3])
                except:
                    coords.append([np.nan, np.nan, np.nan])
            coords = np.array(coords)
            
            # undistort points
            if undistort_points:
                points = np.array(coords)[:,None,:2]
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                coords[:,0] = np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points]).squeeze()
                coords[:,1] = np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points]).squeeze()

            # For each persons combination, create subsets with "nb_cams_off" cameras excluded
            id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off))
            combinations_with_cams_off = np.array([combination.copy()]*len(id_cams_off))
            for i, id in enumerate(id_cams_off):
                combinations_with_cams_off[i,id] = np.nan

            # Try all subsets
            error_comb_all, comb_all, Q_comb_all = [], [], []
            for comb in combinations_with_cams_off:
                error_comb, comb, Q_comb = triangulate_comb(comb, coords, projection_matrices, calib_params, config_dict)
                error_comb_all.append(error_comb)
                comb_all.append(comb)
                Q_comb_all.append(Q_comb)

            error_min = np.nanmin(error_comb_all)
            comb_error_min = [comb_all[np.argmin(error_comb_all)]]
            Q_kpt = [Q_comb_all[np.argmin(error_comb_all)]]
            if error_min < error_threshold_tracking:
                break 

        nb_cams_off += 1
    
    return error_min, comb_error_min, Q_kpt


def read_json(js_file):
    '''
    Read OpenPose json file
    '''
    try:
        with open(js_file, 'r') as json_f:
            js = json.load(json_f)
            json_data = []
            for people in range(len(js['people'])):
                if len(js['people'][people]['pose_keypoints_2d']) < 3: continue
                else:
                    json_data.append(js['people'][people]['pose_keypoints_2d'])
    except:
        json_data = []
    return json_data


def compute_rays(json_coord, calib_params, cam_id):
    '''
    Plucker coordinates of rays from camera to each joint of a person
    Plucker coordinates: camera to keypoint line direction (size 3) 
                         moment: origin ^ line (size 3)
                         additionally, confidence

    INPUTS:
    - json_coord: x, y, likelihood for a person seen from a camera (list of 3*joint_nb)
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cam_id: camera id (int)

    OUTPUT:
    - plucker: array. nb joints * (6 plucker coordinates + 1 likelihood)
    '''

    x = json_coord[0::3]
    y = json_coord[1::3]
    likelihood = json_coord[2::3]
    
    inv_K = calib_params['inv_K'][cam_id]
    R_mat = calib_params['R_mat'][cam_id]
    T = calib_params['T'][cam_id]

    cam_center = -R_mat.T @ T
    plucker = []
    for i in range(len(x)):
        q = np.array([x[i], y[i], 1])
        norm_Q = R_mat.T @ (inv_K @ q -T)
        
        line = norm_Q - cam_center
        norm_line = line/np.linalg.norm(line)
        moment = np.cross(cam_center, norm_line)
        plucker_i = np.concatenate([norm_line, moment, [likelihood[i]]])
        if not np.isnan(plucker_i).any():
            plucker.append(plucker_i)
        else:
            plucker.append(np.array([0.0]*7))

    return np.array(plucker)


def broadcast_line_to_line_distance(p0, p1):
    '''
    Compute the distance between two lines in 3D space.

    see: https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    p0 = (l0,m0), p1 = (l1,m1)
    dist = | (l0,m0) * (l1,m1) | / || l0 x l1 ||
    (l0,m0) * (l1,m1) = l0 @ m1 + m0 @ l1 (reciprocal product)
    
    No need to divide by the norm of the cross product of the directions, since we
    don't need the actual distance but whether the lines are close to intersecting or not
    => dist = | (l0,m0) * (l1,m1) |

    INPUTS:
    - p0: array(nb_persons_detected * 1 * nb_joints * 7 coordinates)
    - p1: array(1 * nb_persons_detected * nb_joints * 7 coordinates)

    OUTPUT:
    - dist: distances between the two lines (not normalized). 
            array(nb_persons_0 * nb_persons_1 * nb_joints)
    '''

    product = np.sum(p0[..., :3] * p1[..., 3:6], axis=-1) + np.sum(p1[..., :3] * p0[..., 3:6], axis=-1)
    dist = np.abs(product)

    return dist


def compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=0.1):
    '''
    Compute the affinity between all the people in the different views.

    The affinity is defined as 1 - distance/max_distance, with distance the
    distance between epipolar lines in each view (reciprocal product of Plucker 
    coordinates).

    Another approach would be to project one epipolar line onto the other camera
    plane and compute the line to point distance, but it is more computationally 
    intensive (simple dot product vs. projection and distance calculation). 
    
    INPUTS:
    - all_json_data_f: list of json data. For frame f, nb_views*nb_persons*(x,y,likelihood)*nb_joints
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cum_persons_per_view: cumulative number of persons per view
    - reconstruction_error_threshold: maximum distance between epipolar lines to consider a match

    OUTPUT:
    - affinity: affinity matrix between all the people in the different views. 
                (nb_views*nb_persons_per_view * nb_views*nb_persons_per_view)
    '''

    # Compute plucker coordinates for all keypoints for each person in each view
    # pluckers_f: dims=(camera, person, joint, 7 coordinates)
    pluckers_f = []
    for cam_id, json_cam  in enumerate(all_json_data_f):
        pluckers = []
        for json_coord in json_cam:
            plucker = compute_rays(json_coord, calib_params, cam_id) # LIMIT TO 15 JOINTS? json_coord[:15*3]
            pluckers.append(plucker)
        pluckers = np.array(pluckers)
        pluckers_f.append(pluckers)

    # Compute affinity matrix
    distance = np.zeros((cum_persons_per_view[-1], cum_persons_per_view[-1])) + 2*reconstruction_error_threshold
    for compared_cam0, compared_cam1 in it.combinations(range(len(all_json_data_f)), 2):
        # skip when no detection for a camera
        if cum_persons_per_view[compared_cam0] == cum_persons_per_view[compared_cam0+1] \
            or cum_persons_per_view[compared_cam1] == cum_persons_per_view[compared_cam1 +1]:
            continue

        # compute distance
        p0 = pluckers_f[compared_cam0][:,None] # add coordinate on second dimension
        p1 = pluckers_f[compared_cam1][None,:] # add coordinate on first dimension
        dist = broadcast_line_to_line_distance(p0, p1)
        likelihood = np.sqrt(p0[..., -1] * p1[..., -1])
        mean_weighted_dist = np.sum(dist*likelihood, axis=-1)/(1e-5 + likelihood.sum(axis=-1)) # array(nb_persons_0 * nb_persons_1)
        
        # populate distance matrix
        distance[cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1], \
                 cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1]] \
                 = mean_weighted_dist
        distance[cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1], \
                 cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1]] \
                 = mean_weighted_dist.T

    # compute affinity matrix and clamp it to zero when distance > reconstruction_error_threshold
    distance[distance > reconstruction_error_threshold] = reconstruction_error_threshold
    affinity = 1 - distance / reconstruction_error_threshold

    return affinity


def circular_constraint(cum_persons_per_view):
    '''
    A person can be matched only with themselves in the same view, and with any 
    person from other views

    INPUT:
    - cum_persons_per_view: cumulative number of persons per view

    OUTPUT:
    - circ_constraint: circular constraint matrix
    '''

    circ_constraint = np.identity(cum_persons_per_view[-1])
    for i in range(len(cum_persons_per_view)-1):
        circ_constraint[cum_persons_per_view[i]:cum_persons_per_view[i+1], cum_persons_per_view[i+1]:cum_persons_per_view[-1]] = 1
        circ_constraint[cum_persons_per_view[i+1]:cum_persons_per_view[-1], cum_persons_per_view[i]:cum_persons_per_view[i+1]] = 1
    
    return circ_constraint


def SVT(matrix, threshold):
    '''
    Find a low-rank approximation of the matrix using Singular Value Thresholding.

    INPUTS:
    - matrix: matrix to decompose
    - threshold: threshold for singular values

    OUTPUT:
    - matrix_thresh: low-rank approximation of the matrix
    '''
    
    U, s, Vt = np.linalg.svd(matrix) # decompose matrix
    s_thresh = np.maximum(s - threshold, 0) # set smallest singular values to zero
    matrix_thresh = U @ np.diag(s_thresh) @ Vt # recompose matrix

    return matrix_thresh


def matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1):
    '''
    Find low-rank approximation of 'affinity' while satisfying the circular constraint.

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - cum_persons_per_view: cumulative number of persons per view
    - circ_constraint: circular constraint matrix
    - max_iter: maximum number of iterations
    - w_rank: threshold for singular values
    - tol: tolerance for convergence
    - w_sparse: regularization parameter

    OUTPUT:
    - new_aff: low-rank approximation of the affinity matrix
    '''

    new_aff = affinity.copy()
    N = new_aff.shape[0]
    index_diag = np.arange(N)
    new_aff[index_diag, index_diag] = 0.
    # new_aff = (new_aff + new_aff.T)/2 # symmetric by construction

    Y = np.zeros_like(new_aff) # Initial deviation matrix / residual ()
    W = w_sparse - new_aff # Initial sparse matrix / regularization (prevent overfitting)
    mu = 64 # initial step size

    for iter in range(max_iter):
        new_aff0 = new_aff.copy()
        
        Q = new_aff + Y*1.0/mu
        Q = SVT(Q,w_rank/mu)
        new_aff = Q - (W + Y)/mu

        # Project X onto dimGroups
        for i in range(len(cum_persons_per_view) - 1):
            ind1, ind2 = cum_persons_per_view[i], cum_persons_per_view[i + 1]
            new_aff[ind1:ind2, ind1:ind2] = 0
            
        # Reset diagonal elements to one and ensure X is within valid range [0, 1]
        new_aff[index_diag, index_diag] = 1.
        new_aff[new_aff < 0] = 0
        new_aff[new_aff > 1] = 1
        
        # Enforce circular constraint
        new_aff = new_aff * circ_constraint
        new_aff = (new_aff + new_aff.T) / 2 # kept just in case X loses its symmetry during optimization 
        Y = Y + mu * (new_aff - Q)
        
        # Compute convergence criteria: break if new_aff is close enough to Q and no evolution anymore
        pRes = np.linalg.norm(new_aff - Q) / N # primal residual (diff between new_aff and SVT result)
        dRes = mu * np.linalg.norm(new_aff - new_aff0) / N # dual residual (diff between new_aff and previous new_aff)
        if pRes < tol and dRes < tol:
            break
        if pRes > 10 * dRes: mu = 2 * mu
        elif dRes > 10 * pRes: mu = mu / 2

        iter +=1

    return new_aff


def person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation):
    '''
    For each detected person, gives their index for each camera

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - min_cameras_for_triangulation: exclude proposals if less than N cameras see them

    OUTPUT:
    - proposals: 2D array: n_persons * n_cams
    '''

    # index of the max affinity for each group (-1 if no detection)
    proposals = []
    for row in range(affinity.shape[0]):
        proposal_row = []
        for cam in range(len(cum_persons_per_view)-1):
            id_persons_per_view = affinity[row, cum_persons_per_view[cam]:cum_persons_per_view[cam+1]]
            proposal_row += [np.argmax(id_persons_per_view) if (len(id_persons_per_view)>0 and max(id_persons_per_view)>0) else -1]
        proposals.append(proposal_row)
    proposals = np.array(proposals, dtype=float)

    # remove duplicates and order
    proposals, nb_detections = np.unique(proposals, axis=0, return_counts=True)
    proposals = proposals[np.argsort(nb_detections)[::-1]]

    # remove row if any value is the same in previous rows at same index (nan!=nan so nan ignored)
    proposals[proposals==-1] = np.nan
    mask = np.ones(proposals.shape[0], dtype=bool)
    for i in range(1, len(proposals)):
        mask[i] = ~np.any(proposals[i] == proposals[:i], axis=0).any()
    proposals = proposals[mask]

    # remove identifications if less than N cameras see them
    nb_cams_per_person = [np.count_nonzero(~np.isnan(p)) for p in proposals]
    proposals = np.array([p for (n,p) in zip(nb_cams_per_person, proposals) if n >= min_cameras_for_triangulation])

    return proposals


def rewrite_json_files(json_tracked_files_f, json_files_f, proposals, n_cams):
    '''
    Write new json files with correct association of people across cameras.

    INPUTS:
    - json_tracked_files_f: list of strings: json files to write
    - json_files_f: list of strings: json files to read
    - proposals: 2D array: n_persons * n_cams
    - n_cams: int: number of cameras

    OUTPUT:
    - json files with correct association of people across cameras
    '''

    for cam in range(n_cams):
        try:
            with open(json_tracked_files_f[cam], 'w') as json_tracked_f:
                with open(json_files_f[cam], 'r') as json_f:
                    js = json.load(json_f)
                    js_new = js.copy()
                    js_new['people'] = []
                    for new_comb in proposals:
                        if not np.isnan(new_comb[cam]):
                            js_new['people'] += [js['people'][int(new_comb[cam])]]
                        else:
                            js_new['people'] += [{}]
                json_tracked_f.write(json.dumps(js_new))
        except:
            if os.path.exists(json_tracked_files_f[cam]):
                os.remove(json_tracked_files_f[cam])


def recap_tracking(config_dict, error=0, nb_cams_excluded=0):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe

    OUTPUT:
    - Message in console
    '''
    
    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    multi_person = config_dict.get('project').get('multi_person')
    likelihood_threshold_association = config_dict.get('personAssociation').get('likelihood_threshold_association')
    tracked_keypoint = config_dict.get('personAssociation').get('single_person').get('tracked_keypoint')
    error_threshold_tracking = config_dict.get('personAssociation').get('single_person').get('reproj_error_threshold_association')
    reconstruction_error_threshold = config_dict.get('personAssociation').get('multi_person').get('reconstruction_error_threshold')
    min_affinity = config_dict.get('personAssociation').get('multi_person').get('min_affinity')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    
    if not multi_person:
        logging.info('\nSingle-person analysis selected.')
        # Error
        mean_error_px = np.around(np.nanmean(error), decimals=1)
        
        calib = toml.load(calib_file)
        cal_keys = [c for c in calib.keys() 
                    if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
                    and isinstance(calib[c],dict)]
        calib_cam1 = calib[cal_keys[0]]
        fm = calib_cam1['matrix'][0][0]
        Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])
        mean_error_mm = np.around(mean_error_px * Dm / fm * 1000, decimals=1)
        
        # Excluded cameras
        mean_cam_off_count = np.around(np.mean(nb_cams_excluded), decimals=2)

        # Recap
        logging.info(f'\n--> Mean reprojection error for {tracked_keypoint} point on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
        logging.info(f'--> In average, {mean_cam_off_count} cameras had to be excluded to reach the demanded {error_threshold_tracking} px error threshold after excluding points with likelihood below {likelihood_threshold_association}.')
    
    else:
        logging.info('\nMulti-person analysis selected.')
        logging.info(f'\n--> A person was reconstructed if the lines from cameras to their keypoints intersected within {reconstruction_error_threshold} m and if the calculated affinity stayed below {min_affinity} after excluding points with likelihood below {likelihood_threshold_association}.')
        logging.info(f'--> Beware that people were sorted across cameras, but not across frames. This will be done in the triangulation stage.')

    logging.info(f'\nTracked json files are stored in {os.path.realpath(poseTracked_dir)}.')
    

def update_frame(cap, ax, slider, text_box, fig):
    '''
    Update the frame display based on slider value
    '''
    frame_num = int(slider.val)
    if isinstance(cap, cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
    else:
        frame = cv2.imread(cap[frame_num])
    ax.images[0].set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text_box.set_val(str(frame_num))
    fig.canvas.draw_idle()

def on_select(event, selected_frames, slider, ax, plt, btn_select):
    '''
    Handle frame selection button click
    '''
    selected_frames.append(int(slider.val))
    if len(selected_frames) == 2:
        plt.close()
    elif len(selected_frames) == 1:
        ax.set_title(f'{ax.get_title().split(":")[0]}: Now select end frame', fontsize=10, color='black', pad=15)
        btn_select.label.set_text('End Frame')

def on_reset(event, selected_frames, slider, ax, btn_select):
    '''
    Handle reset button click
    '''
    selected_frames.clear()
    ax.set_title(f'{ax.get_title().split(":")[0]}: Select start and end frames', fontsize=10, color='black', pad=15)
    btn_select.label.set_text('Start Frame')

def get_frame_range(vid_or_img_files, cam_names, ref_cam):
    '''
    Allows the user to select the frame range by visualizing the video frames from the reference camera.
    
    INPUTS:
    - vid_or_img_files: list of str. Paths to the video files for each camera or to the image directories for each camera.
    - cam_names: list of str. Names of the cameras.
    - ref_cam: str. Name of the reference camera.
    
    OUTPUT:
    - frame_range: list of int. [start_frame, end_frame]
    '''
    logging.info('Manual frame range selection mode: Select the start and end frames.')
    
    try: # video files
        video_files_dict = {cam_name: file for cam_name, file in zip(cam_names, vid_or_img_files)}
    except: # image directories 
        video_files_dict = {cam_name: files for cam_name, files in zip(cam_names, vid_or_img_files)}
    
    selected_frames = []
    
    # Only use reference camera
    vid_or_img_files_cam = video_files_dict.get(ref_cam)
    if not vid_or_img_files_cam:
        logging.warning(f'No video file nor image directory found for reference camera {ref_cam}')
        return [0, 0]  # Return default range if reference camera not found
        
    try:
        cap = cv2.VideoCapture(vid_or_img_files_cam)
        if not cap.isOpened():
            raise
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        cap = vid_or_img_files_cam
        total_frames = len(cap)
    
    # Read first frame
    if isinstance(cap, cv2.VideoCapture):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f'Cannot read frame from video {vid_or_img_files_cam}')
            return [0, total_frames]
    else:
        try:
            frame = cv2.imread(cap[0])
        except:
            logging.warning(f'Cannot read frame from directory {vid_or_img_files_cam}')
            return [0, total_frames]
    
    frame_height, frame_width = frame.shape[:2]
    fig_width, fig_height = frame_width / 200, frame_height / 250
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f'{ref_cam}: Select start and end frames', fontsize=10, color='black', pad=15)
    ax.axis('off')
    
    # Add slider
    ax_slider = plt.axes([ax.get_position().x0, 0.05, ax.get_position().width - 0.112, 0.05])
    slider = Slider(ax_slider, '', 0, total_frames - 1, valfmt='%d')
    
    # Add text box for frame number
    ax_text = plt.axes([ax.get_position().x0 + ax.get_position().width - 0.087, 0.06, 0.04, 0.03])
    text_box = TextBox(ax_text, 'Frame', initial='0')
    
    # Add buttons
    ax_prev = plt.axes([ax.get_position().x0 + ax.get_position().width - 0.044, 0.06, 0.02, 0.03])
    ax_next = plt.axes([ax.get_position().x0 + ax.get_position().width - 0.021, 0.06, 0.02, 0.03])
    btn_prev = plt.Button(ax_prev, '<')
    btn_next = plt.Button(ax_next, '>')
    
    # Add select button with initial text 'Start Frame'
    ax_select = plt.axes([ax.get_position().x0 + ax.get_position().width - 0.1, 0.01, 0.1, 0.03])
    btn_select = plt.Button(ax_select, 'Start Frame')
    
    # Add reset button
    ax_reset = plt.axes([ax.get_position().x0, 0.01, 0.05, 0.03])
    btn_reset = plt.Button(ax_reset, 'Reset')
    
    # Connect callbacks
    slider.on_changed(lambda val: update_frame(cap, ax, slider, text_box, fig))
    text_box.on_submit(lambda text: slider.set_val(int(text)) if text.isdigit() else None)
    btn_prev.on_clicked(lambda event: slider.set_val(max(0, int(slider.val) - 1)))
    btn_next.on_clicked(lambda event: slider.set_val(min(total_frames - 1, int(slider.val) + 1)))
    btn_select.on_clicked(lambda event: on_select(event, selected_frames, slider, ax, plt, btn_select))
    btn_reset.on_clicked(lambda event: on_reset(event, selected_frames, slider, ax, btn_select))
    
    plt.show()
    if isinstance(cap, cv2.VideoCapture):
        cap.release()
    
    if len(selected_frames) != 2:
        logging.warning('Frame range selection incomplete. Using default range.')
        return [0, total_frames]
    
    return sorted(selected_frames)

def select_person_manually(people, image_size):
    fig, ax = plt.subplots()
    person_patches = []

    height, width = image_size

    for i, person in enumerate(people):
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        x_data = keypoints[:, 0]
        y_data = keypoints[:, 1]
        valid = (x_data != 0) & (y_data != 0)
        x_data = x_data[valid]
        y_data = y_data[valid]
        scat = ax.scatter(x_data, -y_data, label=f'Person {i+1}')
        ax.annotate(f'{i+1}', xy=(np.mean(x_data), -np.mean(y_data)), color='red', fontsize=12)
        person_patches.append((scat, i))

    ax.set_title('Click on the person you want to track or close the window to enter ID manually')
    ax.set_xlim([0, width])
    ax.set_ylim([-height, 0])

    selected_person_idx = []

    def onclick(event):
        if event.inaxes == ax:
            x_click = event.xdata
            y_click = event.ydata
            min_dist = float('inf')
            selected_idx = None
            for scat, idx in person_patches:
                x_data = scat.get_offsets()[:, 0]
                y_data = scat.get_offsets()[:, 1]
                distances = np.sqrt((x_data - x_click)**2 + (y_data - y_click)**2)
                if len(distances) > 0:
                    dist = np.min(distances)
                    if dist < min_dist:
                        min_dist = dist
                        selected_idx = idx
            if selected_idx is not None:
                selected_person_idx.append(selected_idx)
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if not selected_person_idx:  # If no person was selected by clicking
        while True:
            print(f"\nAvailable person IDs: {list(range(1, len(people) + 1))}")
            try:
                selected_id = int(input("Enter the ID of the person you want to track (or 0 to cancel): "))
                if selected_id == 0:
                    return None
                if 1 <= selected_id <= len(people):
                    return selected_id - 1
                else:
                    print("Invalid ID. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        return selected_person_idx[0]
    
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_data(post_tracking, json_files, json_folder):
    '''
    Write new json files with tracked person data.

    INPUTS:
    - post_tracking: array of tracked person data
    - json_files: list of json file names
    - json_folder: path to the original json folder

    OUTPUT:
    - save_folder: path to the folder where tracked files are saved
    '''
    base_folder = os.path.dirname(json_folder)
    folder_name = os.path.basename(json_folder)
    save_folder = os.path.join(base_folder, '..', 'pose-associated', folder_name)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, file_name in enumerate(json_files):
        try:
            frame_data = {
                "version": 1.3,
                "people": [{
                    "person_id": [-1],
                    "pose_keypoints_2d": post_tracking[i].tolist(),
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }]
            }
            with open(os.path.join(save_folder, file_name), 'w') as f:
                json.dump(frame_data, f)
        except:
            if os.path.exists(os.path.join(save_folder, file_name)):
                os.remove(os.path.join(save_folder, file_name))
    
    return save_folder

def track_person(folder_folder, image_size, f_range):
    # Get all json files and sort them
    json_files = sorted([f for f in os.listdir(folder_folder) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_folder}")
        return None, None, None, None

    # Filter json files based on frame numbers from filenames
    filtered_json_files = []
    for file_name in json_files:
        try:
            frame_num = int(re.split(r'(\d+)', file_name)[-2])
            if f_range[0] <= frame_num < f_range[1]:
                filtered_json_files.append(file_name)
        except (IndexError, ValueError):
            continue
    
    json_files = filtered_json_files
    if not json_files:
        print(f"No files found within the specified frame range: {f_range}")
        return None, None, None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    pre_tracking_data = []
    total_min_avg = 0
    count_min_avg = 0
    keypoint_count = 0

    for i, file_name in tqdm(enumerate(json_files), total=len(json_files), desc="Processing files", ncols=100):
        data = read_json_file(os.path.join(folder_folder, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if i == 0:
            if people:
                keypoint_count = len(people[0]['pose_keypoints_2d'])
                print(f"Detected {keypoint_count} keypoints in the first frame\n")
            else:
                print("No people detected in the first frame")
                keypoint_count = 75  # Default to 25 keypoints * 3 (x, y, confidence)

        current_pos = np.zeros(keypoint_count)

        if not detected and not right_person:
            if not people:
                pos1.append(current_pos)
            else:
                selected_person_idx = select_person_manually(people, image_size)
                if selected_person_idx is not None:
                    current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    detected = True
                    right_person = True
                    print(f"Manually selected person {selected_person_idx + 1}")
                else:
                    print("No person selected for tracking")
                    return None, None, None, None
                
        elif detected and right_person:
            if people:
                mae = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x0, y0 = np.array(data_to_track[::3]), np.array(data_to_track[1::3])
                    x1, y1 = p1[::3], p1[1::3]
                    valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                    if valid.size == 0:
                        x_mae, y_mae = float('inf'), float('inf')
                    else:
                        x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                        y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                    mae.append(np.mean([x_mae, y_mae]))
                min_avg, I1 = min((val, idx) for (idx, val) in enumerate(mae))
                if min_avg > 100:
                    detected = False
                    right_person = False
                else:
                    current_pos = np.array(people[I1]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    total_min_avg += min_avg
                    count_min_avg += 1
            else:
                detected = False
                right_person = False

        # Adjust current_pos length if keypoint_count changes
        if len(current_pos) < keypoint_count:
            current_pos = np.pad(current_pos, (0, keypoint_count - len(current_pos)), 'constant')
        elif len(current_pos) > keypoint_count:
            keypoint_count = len(current_pos)
            # Pad previous frames' data to match new keypoint_count
            pos1 = [np.pad(p, (0, keypoint_count - len(p)), 'constant') for p in pos1]

        pos1.append(current_pos)

    avg_min_avg = total_min_avg / count_min_avg if count_min_avg > 0 else 0
    print(f"\nAverage min_avg: {avg_min_avg:.2f}")

    return np.array(pos1), json_files, pre_tracking_data, avg_min_avg



def animate_pre_post_tracking(pre_tracking_data, post_tracking_data, folder_name, frame_step=10, interval=100):
    fig, (pre_ax, post_ax) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Add buttons
    replay_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    next_ax = plt.axes([0.91, 0.02, 0.08, 0.04])
    replay_button = plt.Button(replay_ax, 'Replay')
    next_button = plt.Button(next_ax, 'Next')
    
    # Animation control variables
    animation_running = {'value': True}
    
    # Define post-tracking color
    post_color = 'red'
    
    def update(frame):
        pre_ax.clear()
        post_ax.clear()
        pre_ax.set_title(f'Pre-Tracking: {folder_name}')
        pre_ax.set_xlim([0, 4000])
        pre_ax.set_ylim([-3000, 0])
        post_ax.set_title(f'Post-Tracking: {folder_name}')
        post_ax.set_xlim([0, 4000])
        post_ax.set_ylim([-3000, 0])

        # Pre-tracking data
        if frame < len(pre_tracking_data):
            people = pre_tracking_data[frame]
            for person_idx, person in enumerate(people):
                try:
                    # Convert to numpy arrays and ensure float type
                    x_data = np.array(person['pose_keypoints_2d'][0::3], dtype=float)
                    y_data = np.array(person['pose_keypoints_2d'][1::3], dtype=float)
                    
                    # Create mask for valid points
                    valid_x = ~np.isnan(x_data) & ~np.isinf(x_data) & (x_data != 0)
                    valid_y = ~np.isnan(y_data) & ~np.isinf(y_data) & (y_data != 0)
                    valid_mask = valid_x & valid_y
                    
                    x_valid = x_data[valid_mask]
                    y_valid = y_data[valid_mask]
                    
                    if len(x_valid) >= 3:  # Need at least 3 points for ConvexHull
                        valid_points = np.column_stack((x_valid, y_valid))
                        try:
                            hull = ConvexHull(valid_points)
                            # Create scatter plot to get consistent color
                            scat = pre_ax.scatter(x_valid, -y_valid, s=25)
                            color = scat.get_facecolor()[0]
                            # Remove scatter and replot with consistent style
                            scat.remove()
                            # Plot points
                            pre_ax.plot(x_valid, -y_valid, 'o', color=color)
                            # Plot hull edges
                            for simplex in hull.simplices:
                                pre_ax.plot(valid_points[simplex, 0], -valid_points[simplex, 1], '-', color=color)
                            # Fill hull with transparent color
                            hull_points = valid_points[hull.vertices]
                            pre_ax.fill(hull_points[:, 0], -hull_points[:, 1], color=color, alpha=0.2)
                            # Add person number
                            pre_ax.annotate(f'{person_idx}', xy=(np.mean(x_valid), -np.mean(y_valid)), 
                                          color='red', fontsize=12)
                        except:
                            # If ConvexHull fails, just plot the points
                            scat = pre_ax.scatter(x_valid, -y_valid, s=25)
                            color = scat.get_facecolor()[0]
                            scat.remove()
                            pre_ax.plot(x_valid, -y_valid, 'o', color=color)
                except Exception as e:
                    pass  # Skip this person if there's an error

        # Post-tracking data
        if frame < post_tracking_data.shape[0]:
            try:
                # Convert to numpy arrays and ensure float type
                x_data = np.array(post_tracking_data[frame, 0::3], dtype=float)
                y_data = np.array(post_tracking_data[frame, 1::3], dtype=float)
                
                # Create mask for valid points
                valid_x = ~np.isnan(x_data) & ~np.isinf(x_data) & (x_data != 0)
                valid_y = ~np.isnan(y_data) & ~np.isinf(y_data) & (y_data != 0)
                valid_mask = valid_x & valid_y
                
                x_valid = x_data[valid_mask]
                y_valid = y_data[valid_mask]
                
                if len(x_valid) >= 3:  # Need at least 3 points for ConvexHull
                    valid_points = np.column_stack((x_valid, y_valid))
                    try:
                        hull = ConvexHull(valid_points)
                        # Plot points with orange color
                        post_ax.plot(x_valid, -y_valid, 'o', color=post_color)
                        # Plot hull edges
                        for simplex in hull.simplices:
                            post_ax.plot(valid_points[simplex, 0], -valid_points[simplex, 1], '-', color=post_color)
                        # Fill hull with transparent color
                        hull_points = valid_points[hull.vertices]
                        post_ax.fill(hull_points[:, 0], -hull_points[:, 1], color=post_color, alpha=0.2)
                    except:
                        # If ConvexHull fails, just plot the points
                        post_ax.plot(x_valid, -y_valid, 'o', color=post_color)
            except Exception as e:
                pass  # Skip this frame if there's an error

        return pre_ax.get_children() + post_ax.get_children()  # Return artists for animation

    def replay(event):
        ani.frame_seq = ani.new_frame_seq()
        ani.event_source.start()
    
    def next_camera(event):
        animation_running['value'] = False
        plt.close(fig)
    
    replay_button.on_clicked(replay)
    next_button.on_clicked(next_camera)

    max_frames = max(len(pre_tracking_data), post_tracking_data.shape[0])
    frames = list(range(0, max_frames, frame_step))
    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=True)
    
    plt.tight_layout()
    plt.show(block=True)  # Changed to block=True to wait for user input
    
    return animation_running['value']

def associate_all(config_dict):
    '''
    For each frame,
    - Find all possible combinations of detected persons
    - Triangulate 'tracked_keypoint' for all combinations
    - Reproject the point on all cameras
    - Take combination with smallest reprojection error
    - Write json file with only one detected person
    Print recap message
    
    INPUTS: 
    - a calibration file (.toml extension)
    - json files from each camera folders with several detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - json files for each camera with only one person of interest    
    '''
    
    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    multi_person = config_dict.get('project').get('multi_person')
    pose_model = config_dict.get('pose').get('pose_model')
    tracked_keypoint = config_dict.get('personAssociation').get('single_person').get('tracked_keypoint')
    min_cameras_for_triangulation = config_dict.get('triangulation').get('min_cameras_for_triangulation')
    reconstruction_error_threshold = config_dict.get('personAssociation').get('multi_person').get('reconstruction_error_threshold')
    min_affinity = config_dict.get('personAssociation').get('multi_person').get('min_affinity')
    frame_range = config_dict.get('project').get('frame_range')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    use_ConvexHull = config_dict.get('personAssociation').get('use_ConvexHull')
    
    try:
        calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    except:
        raise Exception(f'No .toml calibration direcctory found.')
    try:
        calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    except:
        raise Exception(f'No .toml calibration file found in the {calib_dir}.')
    pose_dir = os.path.join(project_dir, 'pose')
    poseSync_dir = os.path.join(project_dir, 'pose-sync')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')

    # projection matrix from toml calibration file
    P_all = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
        
    # selection of tracked keypoint id
    try: # from skeletons.py
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        else:
            raise ValueError(f"Invalid model_type: {pose_model}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'. Use another network (MMPose, DeepLabCut, OpenPose, AlphaPose, BlazePose...) and convert the output files if you need another model. See documentation.")
   
        model = eval(pose_model)
        
    except:
        try: # from Config.toml
            model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('Model not found in skeletons.py nor in Config.toml')
    tracked_keypoint_id = [node.id for _, _, node in RenderTree(model) if node.name==tracked_keypoint][0]
    
    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    try:
        pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]

    # Try to load from poseSync_dir first, fallback to pose_dir if not found
    try: 
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseSync_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        json_base_dir = poseSync_dir
    except:
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        json_base_dir = pose_dir
        if not any(json_files_names):
            raise ValueError(f'No json files found in {pose_dir} nor {poseSync_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    
    json_files_names = [sort_stringlist_by_last_number(j) for j in json_files_names]

    # reference camera - use camera with minimum number of frames as reference
    min_frame_range = min([len(j) for j in json_files_names])
    ref_cam = [i for i, j in enumerate(json_files_names) if len(j) == min_frame_range][0]
    ref_cam_name = json_dirs_names[ref_cam]  # Get reference camera folder name

    # 2d-pose-associated files creation
    if not os.path.exists(poseTracked_dir): os.mkdir(poseTracked_dir)   
    try: [os.mkdir(os.path.join(poseTracked_dir,k)) for k in json_dirs_names]
    except: pass
    
    error_min_tot, cameras_off_tot = [], []
    
    video_dir = os.path.join(project_dir, 'videos')
    vid_or_img_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    image_size = None
    video_sources = {}  # Dictionary to store video sources for each camera
    
    if not vid_or_img_files:  # If no video files found, try using image directories
        try:
            image_folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
            vid_or_img_files = []
            for image_folder in image_folders:
                img_files = glob.glob(os.path.join(video_dir, image_folder, '*'+vid_img_extension))
                if img_files:  # Only add if directory contains matching files
                    sorted_files = sorted(img_files)
                    vid_or_img_files.append(sorted_files)
                    video_sources[image_folder] = sorted_files  # Store image files for this camera
                    # Get image size from first image that can be read
                    if image_size is None:
                        for img_file in img_files:
                            img = cv2.imread(img_file)
                            if img is not None:
                                image_size = img.shape[:2]  # (height, width)
                                break
        except Exception as e:
            logging.warning(f'Error reading image directories: {str(e)}')
            
        if not vid_or_img_files:
            logging.warning(f'No {vid_img_extension} files found in the image directories.')
    else:
        # Store video files for each camera
        for video_file in vid_or_img_files:
            camera_name = os.path.basename(os.path.dirname(video_file))
            video_sources[camera_name] = video_file
            
        # Get image size from first video that can be read
        for video_file in vid_or_img_files:
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    image_size = frame.shape[:2]  # (height, width)
                    cap.release()
                    break
                cap.release()
        
        if image_size is None:
            logging.warning('Could not read any video files to determine image size.')

    logging.info(f'Using image size: {image_size}')

    # Filter json files based on frame range
    if frame_range == 'manual':
        f_range = get_frame_range(vid_or_img_files, json_dirs_names, ref_cam_name)
    else:
        f_range = [[0,max([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    n_cams = len(json_dirs_names)
    json_folders = [os.path.join(json_base_dir, json_dirs_names[c]) for c in range(n_cams)]

    if use_ConvexHull:
        all_avg_min_avg = []

        for json_folder in json_folders:
                print(f"Tracking using Convex Hull")
                post_tracking, json_files, pre_tracking_data, avg_min_avg = track_person(json_folder, image_size, f_range)

                if post_tracking is None:
                    print(f"Skipping folder {json_folder} due to tracking failure")
                    continue
                
                all_avg_min_avg.append(avg_min_avg)
                animate_pre_post_tracking(pre_tracking_data, post_tracking, os.path.basename(json_folder), frame_step=50, interval=30)
                save_data(post_tracking, json_files, json_folder)

        print(f"Average min_avg across all trials: {np.mean(all_avg_min_avg):.2f}")

    else:
        # Check that camera number is consistent between calibration file and pose folders
        if n_cams != len(P_all):
            raise Exception(f'Error: The number of cameras is not consistent:\
                        Found {len(P_all)} cameras in the calibration file,\
                        and {n_cams} cameras based on the number of pose folders.')
        
        for f in tqdm(range(*f_range)):
            # print(f'\nFrame {f}:')
            json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)',j)[-2])==f] for c in range(n_cams)]
            json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
            try:
                json_files_f = [os.path.join(poseSync_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]
                with open(os.path.exist(json_files_f[0])) as json_exist_test: pass
            except:
                json_files_f = [os.path.join(pose_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]
            json_tracked_files_f = [os.path.join(poseTracked_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]



            if not multi_person:
                    # all possible combinations of persons
                    personsIDs_comb = persons_combinations(json_files_f) 
                    
                    # choose persons of interest and exclude cameras with bad pose estimation
                    error_proposals, proposals, Q_kpt = best_persons_and_cameras_combination(config_dict, json_files_f, personsIDs_comb, P_all, tracked_keypoint_id, calib_params)

                    if not np.isinf(error_proposals):
                        error_min_tot.append(np.nanmean(error_proposals))
                    cameras_off_count = np.count_nonzero([np.isnan(comb) for comb in proposals]) / len(proposals)
                    cameras_off_tot.append(cameras_off_count)            

            else:
                    # read data
                    all_json_data_f = []
                    for js_file in json_files_f:
                        all_json_data_f.append(read_json(js_file))
                    #TODO: remove people with average likelihood < 0.3, no full torso, less than 12 joints... (cf filter2d in dataset/base.py L498)
                    
                    # obtain proposals after computing affinity between all the people in the different views
                    persons_per_view = [0] + [len(j) for j in all_json_data_f]
                    cum_persons_per_view = np.cumsum(persons_per_view)
                    affinity = compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=reconstruction_error_threshold)
                    circ_constraint = circular_constraint(cum_persons_per_view)
                    affinity = affinity * circ_constraint
                    #TODO: affinity without hand, face, feet (cf ray.py L31)
                    affinity = matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1)
                    affinity[affinity<min_affinity] = 0
                    proposals = person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation)
            
            # rewrite json files with a single or multiple persons of interest
            rewrite_json_files(json_tracked_files_f, json_files_f, proposals, n_cams)

    # recap message
    recap_tracking(config_dict, error_min_tot, cameras_off_tot)
    
