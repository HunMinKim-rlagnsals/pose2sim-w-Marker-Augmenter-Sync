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
    reprojection, euclidean_distance, sort_stringlist_by_last_number
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

def onclick_person_select(event, ax, person_patches, selected_person_idx, fig):
    if event.inaxes == ax:
        x_click = event.xdata
        y_click = event.ydata
        min_dist = float('inf')
        selected_idx = None
        for scat, hull_patch, idx in person_patches:
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

def on_motion_person_select(event, ax, person_patches, fig):
    if event.inaxes == ax:
        for scat, hull_patch, idx in person_patches:
            path = hull_patch.get_path()
            if path.contains_point([event.xdata, event.ydata]):
                hull_patch.set_alpha(0.5)
                scat.set_sizes([50])  # Highlight points
            else:
                hull_patch.set_alpha(0.2)
                scat.set_sizes([25])  # Normal size
        fig.canvas.draw_idle()

def select_person_manually(people, frame=None):

    if frame is not None:
        frame_height, frame_width = frame.shape[:2]
        
    fig, ax = plt.subplots(figsize=(12, 8))
    person_patches = []
    hull_patches = []  # Store hull patches for highlighting

    for i, person in enumerate(people):
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        x_data = keypoints[:, 0]
        y_data = keypoints[:, 1]
        valid = (x_data != 0) & (y_data != 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
        x_data = x_data[valid]
        y_data = y_data[valid]

        if len(x_data) < 3:
            continue
            
        # Add ConvexHull first to check if it's possible
        points = np.column_stack((x_data, -y_data))
        hull = ConvexHull(points)
            
        # If ConvexHull creation successful, add scatter and polygon
        scat = ax.scatter(x_data, -y_data, label=f'Person {i}', s=25)
        hull_path = plt.Polygon(points[hull.vertices], alpha=0.2, color=scat.get_facecolor())
        ax.add_patch(hull_path)
        hull_patches.append(hull_path)
            
        ax.annotate(f'{i}', xy=(np.mean(x_data), -np.mean(y_data)), color='red', fontsize=12)
        person_patches.append((scat, hull_path, i))

    if not person_patches:
        logging.error("No valid persons found with enough keypoints for ConvexHull")
        return 0  # Default to first person if no valid ConvexHull could be created

    ax.set_title('Click on the person you want to track')
    ax.set_xlim([0, frame_width])
    ax.set_ylim([-frame_height, 0])

    selected_person_idx = []
    
    fig.canvas.mpl_connect('motion_notify_event', 
                          lambda event: on_motion_person_select(event, ax, person_patches, fig))
    fig.canvas.mpl_connect('button_press_event', 
                          lambda event: onclick_person_select(event, ax, person_patches, selected_person_idx, fig))
    plt.show()

    if not selected_person_idx:
        logging.warning("No person selected, defaulting to person 0")
        return 0
    
    selected_id = selected_person_idx[0]
    logging.info(f"Selected person ID: {selected_id}")
    return selected_id

def track_with_convexhull(json_files_f, selected_id, image_size=None):
    '''
    Track person using ConvexHull method with area and center distance consideration
    Uses specific keypoints (Hip, Knee, Neck, Shoulder, Elbow) for tracking
    '''
    if not json_files_f or 'none' in json_files_f:
        logging.debug("No valid json files provided")
        return None
    
    # Set max_center_dist as half of the maximum image dimension
    max_image_size = max(image_size)
    max_center_dist = max_image_size / 2

    try:
        # Static variables to store last known information
        if not hasattr(track_with_convexhull, 'last_hull_center'):
            track_with_convexhull.last_hull_center = None
            track_with_convexhull.last_hull_area = None
            track_with_convexhull.initial_id = selected_id
            track_with_convexhull.last_keypoints = None  # Store previous frame's keypoint pattern

        # Selected keypoint indices for tracking
        selected_indices = [19,  # Hip
                          12, 11,  # RHip, LHip
                          14, 13,  # RKnee, LKnee
                          18, 17,  # Neck, Head
                          6, 5,  # RShoulder, LShoulder
                          8, 7]  # RElbow, LElbow
        
        with open(json_files_f[0], 'r') as f:
            data = json.load(f)
            people = data.get('people', [])
            
            if not people:
                return None

            # First frame or no previous information
            if track_with_convexhull.last_hull_center is None:
                if selected_id >= len(people):
                    return None

                # Get initial person's information
                keypoints_to_track = np.array(people[selected_id]['pose_keypoints_2d']).reshape(-1, 3)
                # Filter only selected keypoints
                keypoints_to_track = keypoints_to_track[selected_indices]
                valid_points_to_track = keypoints_to_track[~np.isnan(keypoints_to_track).any(axis=1) & 
                                                         (keypoints_to_track != 0).all(axis=1)][:, :2]
                
                if len(valid_points_to_track) < 3:
                    return None

                try:
                    hull_to_track = ConvexHull(valid_points_to_track)
                    track_with_convexhull.last_hull_center = np.mean(valid_points_to_track[hull_to_track.vertices], axis=0)
                    track_with_convexhull.last_hull_area = hull_to_track.area
                    track_with_convexhull.last_keypoints = keypoints_to_track
                    return selected_id
                except Exception as e:
                    return None

            # For subsequent frames, find the best match based on previous frame
            best_score = float('-inf')
            best_id = None
            best_metrics = None

            for idx, person in enumerate(people):
                keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
                # Filter only selected keypoints
                keypoints = keypoints[selected_indices]
                valid_points = keypoints[~np.isnan(keypoints).any(axis=1) & 
                                      (keypoints != 0).all(axis=1)][:, :2]

                if len(valid_points) < 3:
                    continue

                try:
                    hull = ConvexHull(valid_points)
                    hull_center = np.mean(valid_points[hull.vertices], axis=0)
                    hull_area = hull.area

                    # Calculate metrics relative to last known position
                    center_dist = euclidean_distance(track_with_convexhull.last_hull_center, hull_center)
                    area_diff = abs(track_with_convexhull.last_hull_area - hull_area) / max(track_with_convexhull.last_hull_area, hull_area)
                    
                    # Combine metrics into a single score
                    # Weight the different components
                    center_weight = 0.6
                    area_weight = 0.4

                    # Convert center_dist to a similarity score (closer to 1 is better)
                    center_score = max(0, 1 - (center_dist / max_center_dist))
                    
                    # Convert area_diff to a similarity score (closer to 1 is better)
                    area_score = 1 - area_diff
                    
                    # Combine scores
                    score = (center_weight * center_score + 
                            area_weight * area_score )
                    
                    if score > best_score:
                        best_score = score
                        best_id = idx
                        best_metrics = {
                            'center_dist': center_dist,
                            'area_diff': area_diff * 100,
                            'hull_center': hull_center,
                            'hull_area': hull_area,
                            'keypoints': keypoints,
                            'center_score': center_score,
                            'area_score': area_score,
                            'total_score': score
                        }

                except Exception as e:
                    continue

            if best_id is None:
                return None

            # Update last known information for next frame
            track_with_convexhull.last_hull_center = best_metrics['hull_center']
            track_with_convexhull.last_hull_area = best_metrics['hull_area']
            track_with_convexhull.last_keypoints = best_metrics['keypoints']

            return best_id

    except Exception as e:
        return None

# def calculate_keypoint_similarity(prev_keypoints, curr_keypoints, confidence_threshold=0.5, consider_scale=False):
#     """
#     Calculate similarity between two sets of keypoints using Procrustes Analysis
#     크기를 고려할지 여부를 선택할 수 있음
    
#     Args:
#         prev_keypoints: Previous frame keypoints (N x 3 array: x, y, confidence)
#         curr_keypoints: Current frame keypoints (N x 3 array: x, y, confidence)
#         confidence_threshold: Minimum confidence value to consider a keypoint valid
#         consider_scale: Whether to consider scale differences in similarity calculation
        
#     Returns:
#         float: Similarity score between 0 and 1, where 1 means identical
#     """
#     if prev_keypoints is None or curr_keypoints is None:
#         return 0.0
        
#     try:
#         # 사용할 키포인트 인덱스 (Hip, Knee, Neck, Shoulder)
#         selected_indices = [19,  # Hip
#                           12, 11,  # RHip, LHip
#                           14, 13,  # RKnee, LKnee
#                           18, 17,  # Neck, Head
#                           6, 5,  # RShoulder, LShoulder
#                           8, 7] # RElbow, LElbow
        
#         # 선택된 키포인트만 추출
#         prev_selected = prev_keypoints[selected_indices]
#         curr_selected = curr_keypoints[selected_indices]
        
#         # 양쪽 프레임에서 신뢰도가 높은 키포인트만 선택
#         prev_valid_mask = prev_selected[:, 2] > confidence_threshold
#         curr_valid_mask = curr_selected[:, 2] > confidence_threshold

#         # 공통으로 유효한 키포인트만 선택
#         common_valid_mask = prev_valid_mask & curr_valid_mask
        
#         # 유효한 키포인트가 3개 미만이면 유사도 0 반환
#         if np.sum(common_valid_mask) < 3:
#             return 0.0
            
#         # 유효한 키포인트의 좌표만 추출
#         prev_valid = prev_selected[common_valid_mask][:, :2]
#         curr_valid = curr_selected[common_valid_mask][:, :2]
        
#         # 각 키포인트 세트의 중심을 원점으로 이동
#         prev_centered = prev_valid - np.mean(prev_valid, axis=0)
#         curr_centered = curr_valid - np.mean(curr_valid, axis=0)
        
#         if consider_scale:
#             # 크기를 고려하는 경우: 정규화 생략
#             prev_normalized = prev_centered
#             curr_normalized = curr_centered
            
#             # 크기 차이 계산
#             prev_scale = np.linalg.norm(prev_centered, 'fro')
#             curr_scale = np.linalg.norm(curr_centered, 'fro')
#             scale_diff = abs(prev_scale - curr_scale) / max(prev_scale, curr_scale)
            
#         else:
#             # 크기를 고려하지 않는 경우: 이전처럼 정규화 수행
#             prev_norm = np.linalg.norm(prev_centered, 'fro')
#             curr_norm = np.linalg.norm(curr_centered, 'fro')
            
#             if prev_norm == 0 or curr_norm == 0:
#                 return 0.0
                
#             prev_normalized = prev_centered / prev_norm
#             curr_normalized = curr_centered / curr_norm
        
#         # Procrustes Analysis: optimal rotation 계산
#         H = prev_normalized.T @ curr_normalized
#         U, _, Vt = np.linalg.svd(H)
#         R = Vt.T @ U.T
        
#         # Procrustes distance 계산 및 similarity score로 변환
#         aligned_curr = curr_normalized @ R
#         procrustes_distance = np.linalg.norm(prev_normalized - aligned_curr, 'fro')
        
#         if consider_scale:
#             # 크기 차이를 유사도에 반영
#             similarity = (1.0 / (1.0 + procrustes_distance)) * (1.0 - scale_diff)
#         else:
#             similarity = 1.0 / (1.0 + procrustes_distance)
        
#         return similarity
        
#     except Exception as e:
#         logging.debug(f"Error calculating keypoint similarity: {str(e)}")
#         return 0.0

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
    
    if not vid_or_img_files:  # If no video files found, try using image directories
        try:
            image_folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
            vid_or_img_files = []
            for image_folder in image_folders:
                img_files = glob.glob(os.path.join(video_dir, image_folder, '*'+vid_img_extension))
                if img_files:  # Only add if directory contains matching files
                    vid_or_img_files.append(sorted(img_files))
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

    # Handle manual frame range selection
    if frame_range == 'manual':
        # get frame range using reference camera name
        frame_range = get_frame_range(vid_or_img_files, json_dirs_names, ref_cam_name)
    
    f_range = [[0,max([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    n_cams = len(json_dirs_names)

    # Check that camera number is consistent between calibration file and pose folders
    if n_cams != len(P_all):
        raise Exception(f'Error: The number of cameras is not consistent:\
                    Found {len(P_all)} cameras in the calibration file,\
                    and {n_cams} cameras based on the number of pose folders.')
    
    prev_proposals = None  # Store previous frame's proposals
    id_changes = [0] * n_cams  # Track ID changes for each camera
    prev_ids = [None] * n_cams  # Store previous IDs for each camera

    for f in tqdm(range(*f_range)):
        json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)',j)[-2])==f] for c in range(n_cams)]
        json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
        json_files_f = [os.path.join(json_base_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]
        json_tracked_files_f = [os.path.join(poseTracked_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]


        # NOTE!: 이곳까지 OK

        if not multi_person:
            if use_ConvexHull and f == f_range[0]:  # First frame for ConvexHull
                # Get initial person selection for each camera
                initial_selected_ids = []
                for cam in range(n_cams):
                    try:
                        with open(json_files_f[cam], 'r') as f_cam:
                            data = json.load(f_cam)
                            logging.info(f'\nSelecting person for camera {cam+1}')
                            # Get the first frame
                            if isinstance(vid_or_img_files[cam], str):  # video file
                                cap = cv2.VideoCapture(vid_or_img_files[cam])
                                ret, frame = cap.read()
                                cap.release()
                            else:  # image directory
                                frame = cv2.imread(vid_or_img_files[cam][0])
                            initial_selected_id = select_person_manually(data.get('people', []), frame)
                            if initial_selected_id is None:
                                raise ValueError(f"No person selected for tracking in camera {cam+1}")
                            initial_selected_ids.append(initial_selected_id)
                    except:
                        initial_selected_ids.append(0)
                        logging.warning(f"Failed to manually select person for camera {cam+1}, defaulting to person 0")
            
            if use_ConvexHull:
                # Track person using ConvexHull method for each camera independently
                proposals = []
                for cam in range(n_cams):
                    if f == f_range[0]:  # Use initially selected person
                        tracked_id = initial_selected_ids[cam]
                        prev_ids[cam] = tracked_id
                    else:  # Track from previous frame
                        tracked_id = track_with_convexhull([json_files_f[cam]], initial_selected_ids[cam], image_size)
                        prev_ids[cam] = tracked_id
                    proposals.append([tracked_id if tracked_id is not None else np.nan])
                proposals = np.array(proposals).T

                # Compare current proposals with previous frame's proposals
                if prev_proposals is not None:
                    for cam in range(n_cams):
                        curr_id = proposals[0][cam]
                        prev_id = prev_proposals[0][cam]
                        
                        # Check if either current or previous ID is nan
                        curr_is_nan = np.isnan(curr_id) if isinstance(curr_id, float) else False
                        prev_is_nan = np.isnan(prev_id) if isinstance(prev_id, float) else False
                        
                        # Only compare if neither is nan and they are different
                        if not (curr_is_nan or prev_is_nan):  # 둘 다 유효한 ID인 경우만
                            if curr_id != prev_id:  # ID가 변경된 경우
                                # logging.warning(f"Frame {f}, Camera {cam+1}: ID changed from {int(prev_id)} to {int(curr_id)}")
                                id_changes[cam] += 1  # Increment ID change counter
                                prev_ids[cam] = int(curr_id)  # Update previous ID
                        elif curr_is_nan and not prev_is_nan:  # 현재 프레임에서 추적 실패
                            # logging.warning(f"Frame {f}, Camera {cam+1}: Lost tracking. Previous ID was {int(prev_id)}")
                            id_changes[cam] += 1  # Count losing track as an ID change
                        elif not curr_is_nan and prev_is_nan:  # 추적 복구
                            # logging.warning(f"Frame {f}, Camera {cam+1}: Recovered tracking with ID {int(curr_id)}")
                            prev_ids[cam] = int(curr_id)
                            id_changes[cam] += 1  # Count recovering track as an ID change

                # Store current proposals for next frame comparison
                prev_proposals = proposals.copy()

                # Final logging only at the last frame
                if f == f_range[1] - 1:  # Changed from f_range[-1] to f_range[1] - 1
                    logging.info("\n=== Final Tracking Summary ===")
                    for cam in range(n_cams):
                        logging.info(f"\nCamera {json_dirs_names[cam]}:")
                        logging.info(f"  - Initial ID: {initial_selected_ids[cam]}")
                        logging.info(f"  - Final ID: {prev_ids[cam]}")
                        logging.info(f"  - Total ID changes: {id_changes[cam]}")
                        if id_changes[cam] > 0:
                            logging.warning(f"  - Warning: ID changed {id_changes[cam]} times in this camera")
            else:
                # Original single-person tracking logic
                personsIDs_comb = persons_combinations(json_files_f) 
                error_proposals, proposals, Q_kpt = best_persons_and_cameras_combination(config_dict, json_files_f, personsIDs_comb, P_all, tracked_keypoint_id, calib_params)

                if not np.isinf(error_proposals):
                    error_min_tot.append(np.nanmean(error_proposals))
                cameras_off_count = np.count_nonzero([np.isnan(comb) for comb in proposals]) / len(proposals)
                cameras_off_tot.append(cameras_off_count)            

        else:
            # Original multi-person tracking logic
            all_json_data_f = []
            for js_file in json_files_f:
                all_json_data_f.append(read_json(js_file))
            
            persons_per_view = [0] + [len(j) for j in all_json_data_f]
            cum_persons_per_view = np.cumsum(persons_per_view)
            affinity = compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=reconstruction_error_threshold)
            circ_constraint = circular_constraint(cum_persons_per_view)
            affinity = affinity * circ_constraint
            affinity = matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1)
            affinity[affinity<min_affinity] = 0
            proposals = person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation)
        
        # rewrite json files with a single or multiple persons of interest
        rewrite_json_files(json_tracked_files_f, json_files_f, proposals, n_cams)

    # Visualize tracking results for each camera
    for cam in range(n_cams):
        # Load pre-tracking data
        pre_tracking_data = []
        for f in range(*f_range):
            try:
                with open(os.path.join(json_base_dir, json_dirs_names[cam], json_files_names[cam][f]), 'r') as f_pre:
                    pre_tracking_data.append(json.load(f_pre)['people'])
            except:
                pre_tracking_data.append([])

        # Load post-tracking data
        keypoints_array = []
        for f in range(*f_range):
            try:
                with open(os.path.join(poseTracked_dir, json_dirs_names[cam], json_files_names[cam][f]), 'r') as f_post:
                    data = json.load(f_post)['people']
                    if data:  # If there are tracked people
                        keypoints = np.array(data[0]['pose_keypoints_2d'])  # Take the first (tracked) person
                        keypoints_array.append(keypoints)
                    else:
                        keypoints_array.append(np.zeros(len(keypoints)))  # Use zeros if no person detected
            except:
                if keypoints_array:  # If we have seen at least one valid frame
                    keypoints_array.append(np.zeros_like(keypoints_array[0]))
                else:
                    continue

        # Convert to numpy array for post-tracking visualization
        post_tracking_array = np.array(keypoints_array)

        # Animate the comparison
        logging.info(f"\nVisualizing tracking results for camera {json_dirs_names[cam]}")
        animate_pre_post_tracking(pre_tracking_data, post_tracking_array, json_dirs_names[cam])

    # recap message
    recap_tracking(config_dict, error_min_tot, cameras_off_tot)
    
