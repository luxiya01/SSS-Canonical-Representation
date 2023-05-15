from cProfile import label
from operator import le
import string
import sys
from collections import defaultdict
import pandas as pd
import math

from requests import patch

sys.path.append('/home/viki/Master_Thesis/auvlib/scripts')
sys.path.append('/home/viki/Master_Thesis/SSS-Canonical-Representation')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, patch_rotated
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from canonical_transformation import canonical_trans
import cv2 as cv

path1 = '/home/viki/Master_Thesis/SSS-Canonical-Representation/ssh174/ssh174_raw.npy'
path2 = '/home/viki/Master_Thesis/SSS-Canonical-Representation/ssh174/ssh173_raw.npy'

canonical_path1 = '/home/viki/Master_Thesis/SSS-Canonical-Representation/ssh174/ssh174_canonical.npy'
canonical_path2 = '/home/viki/Master_Thesis/SSS-Canonical-Representation/ssh174/ssh173_canonical.npy'
'''
SSH-0170-l01s01-20210210-111341.XTF
SSH-0171-l02s01-20210210-112129.XTF
SSH-0172-l03s01-20210210-112929.XTF
SSH-0173-l04s01-20210210-113741.XTF
SSH-0174-l05s01-20210210-114538.XTF
'''
xtf_file1 = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0174-l05s01-20210210-114538.XTF"
xtf_file2 = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0173-l04s01-20210210-113741.XTF"

draping_res_folder = '/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/9-0169to0182-nbr_pings-5204'
annotation_file = '/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/annotations/SSH-0174/correspondence_annotations_SSH-0174.json'
'''
SSH-0170-l01s01-20210210-111341.cereal
SSH-0171-l02s01-20210210-112129.cereal
SSH-0172-l03s01-20210210-112929.cereal
SSH-0173-l04s01-20210210-113741.cereal
SSH-0174-l05s01-20210210-114538.cereal
'''
filename1 = 'SSH-0174-l05s01-20210210-114538.cereal'
filename2 = 'SSH-0173-l04s01-20210210-113741.cereal'
patch_outpath = '/home/viki/Master_Thesis/SSS-Canonical-Representation/data/ssh170/patch_pairs/ssh171'

# Possible LambertianModel values: tan, tan_square, sin, sin_square
def cano_img_gen(
    path1,
    path2,
    canonical_path1,
    canonical_path2,
    xtf_file1,
    xtf_file2,
    draping_res_folder,
    annotation_file,
    filename1,
    filename2,
    patch_outpath,
    LambertianModel="sin_square",
    deconv=False,
    beam_pattern_correction=False,
    mesh_file="/media/li/LaCie/local-corr/data/210209/pp/EM2040/9-0159toend/mesh-data-roll0.35.cereal_resolution0.5m.npz",
    svp_file="/media/li/LaCie/local-corr/data/210209/pp/processed_svp.txt"
):
    '''
    Generate canonical image and Patch class pairs from given xtf files, annotation and mesh

    Parameters
    ----------
    path1/2: str
        path to save raw sss images as np.array
    canonical_path1/2: str
        path to save canonical sss images as np.array
    xtf_file1/2: str
        path to xtf files
    draping_res_folder: str
        path to draping_res_folder, needed for annotator object
    annotation_file: str
        path to annotation
    filename1/2: str
        path to the annotated images, needed to obtain mutual kps of image pairs
    patch_outpath: str
        path to save patch pairs
    mesh_file: str
        corresponding mesh file of the current dataset
    svp_file: str
        processed_svp file of the current dataset
    '''
    matched_kps1 = []
    matched_kps2 = []
    annotator = SSSFolderAnnotator(draping_res_folder, annotation_file)

    for kp_id, kp_dict in annotator.annotations_manager.correspondence_annotations.items(
    ):
        if filename1 in kp_dict and filename2 in kp_dict:
            matched_kps1.append(list(kp_dict[filename1]))
            matched_kps2.append(list(kp_dict[filename2]))

    matched_kps1 = np.array(matched_kps1).astype(np.float32)
    matched_kps2 = np.array(matched_kps2).astype(np.float32)

    # do canonical transform and save img / kps
    raw_img1, canonical_img1, r1, rg1, rg_bar1, canonical_kps1 = canonical_trans(
        xtf_file1,
        mesh_file,
        svp_file,
        matched_kps1,
        deconv=deconv,
        len_bins=1301,
        LambertianModel=LambertianModel,
        beam_pattern_correction=beam_pattern_correction)
    np.save(path1, raw_img1)
    np.save(canonical_path1, canonical_img1)
    cv_file1 = cv.FileStorage(canonical_path1 + '.xml', cv.FILE_STORAGE_WRITE)
    cv_file1.write("ct_img", canonical_img1)
    cv_file1.release()

    raw_img2, canonical_img2, r2, rg2, rg_bar2, canonical_kps2 = canonical_trans(
        xtf_file2,
        mesh_file,
        svp_file,
        matched_kps2,
        deconv=deconv,
        len_bins=1301,
        LambertianModel=LambertianModel,
        beam_pattern_correction=beam_pattern_correction)
    np.save(path2, raw_img2)
    np.save(canonical_path2, canonical_img2)
    cv_file2 = cv.FileStorage(canonical_path2 + '.xml', cv.FILE_STORAGE_WRITE)
    cv_file2.write("ct_img", canonical_img2)
    cv_file2.release()
    return

    # generate and save patch pairs from raw and cano images
    patch_size = 500
    generate_patches_pair(path1, path2, matched_kps1, matched_kps2, False,
                          patch_size, patch_outpath)
    generate_patches_pair(canonical_path1, canonical_path2, canonical_kps1,
                          canonical_kps2, True, patch_size, patch_outpath)


# Possible LambertianModel values: tan, tan_square, sin, sin_square
def cano_dir_gen(data_path, ref_ssh_idx, LambertianModel="sin_square", deconv=False, beam_pattern_correction=False):
    #data_path = '/home/viki/Master_Thesis/SSS-Canonical-Representation/data/ssh'
    sss_no = ['170', '171', '172', '173', '174']
    xtf_files = [
        'SSH-0170-l01s01-20210210-111341.XTF',
        'SSH-0171-l02s01-20210210-112129.XTF',
        'SSH-0172-l03s01-20210210-112929.XTF',
        'SSH-0173-l04s01-20210210-113741.XTF',
        'SSH-0174-l05s01-20210210-114538.XTF'
    ]
    annotated_files = [
        'SSH-0170-l01s01-20210210-111341.cereal',
        'SSH-0171-l02s01-20210210-112129.cereal',
        'SSH-0172-l03s01-20210210-112929.cereal',
        'SSH-0173-l04s01-20210210-113741.cereal',
        'SSH-0174-l05s01-20210210-114538.cereal'
    ]
    xtf_dir = '/media/li/LaCie/local-corr/data/210209/pp/ETPro/ssh/9-0169to0182/'
    #xtf_dir = '/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/'
    draping_res_folder = '/media/li/LaCie/local-corr/data/210209/pp/ETPro/ssh/9-0169to0182/9-0169to0182-nbr_pings-5204'
    #draping_res_folder = '/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/9-0169to0182-nbr_pings-5204'
    annotation_dir = '/media/li/LaCie/local-corr/data/210209/annotations/Annotated_Area_2/'
    # annotation_dir = '/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/annotations/SSH-0'

    i = ref_ssh_idx
    #for i in range(5):
    parent_path = data_path + sss_no[i] + f'/{LambertianModel}_patch_pairs'
    deconv_img_dir = data_path + sss_no[i] + f'/{LambertianModel}_cano_img'
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    if not os.path.exists(deconv_img_dir):
        os.mkdir(deconv_img_dir)
    #annotation_file = annotation_dir + sss_no[
    #    i] + '/correspondence_annotations_SSH-0' + sss_no[i] + '.json'
    annotation_file = annotation_dir + '/correspondence_annotations_SSH-0' + sss_no[i] + '.json'
    for j in range(5):
        if sss_no[i] == sss_no[j]:
            continue
        patch_outpath = parent_path + '/ssh' + sss_no[j]
        if not os.path.exists(patch_outpath):
            os.mkdir(patch_outpath)
        path1 = deconv_img_dir + '/ssh' + sss_no[i] + '_raw.npy'
        path2 = deconv_img_dir + '/ssh' + sss_no[j] + '_raw.npy'
        canonical_path1 = deconv_img_dir + '/ssh' + sss_no[
            i] + '_canonical.npy'
        canonical_path2 = deconv_img_dir + '/ssh' + sss_no[
            j] + '_canonical.npy'
        xtf_file1 = xtf_dir + xtf_files[i]
        xtf_file2 = xtf_dir + xtf_files[j]
        filename1 = annotated_files[i]
        filename2 = annotated_files[j]

#        if os.path.exists(canonical_path1) and os.path.exists(canonical_path2):
#            print(f'Skipping existing pair ({canonical_path1}, {canonical_path2})')
#            continue
        cano_img_gen(path1, path2, canonical_path1, canonical_path2,
                        xtf_file1, xtf_file2, draping_res_folder,
                        annotation_file, filename1, filename2, patch_outpath,
                        LambertianModel=LambertianModel,
                        deconv=deconv,
                        beam_pattern_correction=beam_pattern_correction)
        print(f'Generate patch pairs in {patch_outpath}')


def desc_evaluation(patch_outpath, matcher, descType, rotate, cano_match,
                    raw_match, similarity_comparision, similarity_comparision_kl, corr_comparison,
                    descriptors_df,
                    fixed_size=False):
    '''
    Conduct evaluation based on desc matching for one set of patch pairs: sss17x-sss17y
    '''
    pkl_file_filter = lambda x: 'fixed_bin' in x if fixed_size else 'fixed_bin' not in x

    pkl_files = [x for x in os.listdir(patch_outpath) if x.split('.')[-1]=='pkl']
    pkl_files = [x for x in pkl_files if pkl_file_filter(x)]
        
    number_of_pair = math.ceil(len(pkl_files)/2)
    #number_of_pair = int(len([x for x in 
    #    os.listdir(patch_outpath) if x.split('.')[-1]=='pkl']) / 2)
    patch_comparision = np.full((number_of_pair, 3), '', dtype=object)
    cano_correct = []
    raw_correct = []
    cano_matched = []
    raw_matched = []

    # get the raw and canonical patch pairs stored in the same patch no
    for filename in os.listdir(patch_outpath):
        if '.pkl' not in filename or not pkl_file_filter(filename):
            continue

        patch_no = filename.split('_')[0]
        if not patch_no in patch_comparision:
            ind = np.argwhere(patch_comparision[:, 0] == '')[0][0]
            patch_comparision[ind, 0] = patch_no
            patch_comparision[ind, 1] = filename
        else:
            ind = np.argwhere(patch_comparision[:, 0] == patch_no)[0][0]
            patch_comparision[ind, 2] = filename

    for i in range(number_of_pair):
        if patch_comparision[i, 1] == '' or patch_comparision[i, 2] == '':
            continue
        with open(os.path.join(patch_outpath, patch_comparision[i, 1]),
                  'rb') as f1:
            patch1 = pickle.load(f1)
        with open(os.path.join(patch_outpath, patch_comparision[i, 2]),
                  'rb') as f2:
            patch2 = pickle.load(f2)

        # if len(patch1.annotated_keypoints1) != len(patch2.annotated_keypoints1):
        #     print(f'Canonical and raw kypoints number not equal!!!')
        #     continue

        if patch1.is_canonical:
            patch_cano = patch1
            patch_raw = patch2
        else:
            patch_cano = patch2
            patch_raw = patch1
        assert patch_cano.patch_id == patch_raw.patch_id

        DescDict = {
            "sift": [cv.SIFT_create(), 0.8],
            "orb": [cv.ORB_create(edgeThreshold=0), 0.8],
            "brisk": [cv.BRISK_create()],
            "freak": [cv.xfeatures2d.FREAK_create()],
            #"surf": [cv.xfeatures2d.SURF_create(), 0.8]
        }
        desc = DescDict.get(descType)

        if not (patch_cano.sss_waterfall_image1.size
                and patch_cano.sss_waterfall_image2.size
                and patch_raw.sss_waterfall_image1.size
                and patch_raw.sss_waterfall_image2.size):
            print(f'Patch img does not exit!!!')
            continue
        canomax = 2.1
        rawmax = 3
        kp_size = 16

        # generate desc
        if not rotate:
            patch_cano_annotated_kps1, patch_cano_desc1, patch_cano_img1_normalized = compute_desc_at_annotated_locations(
                patch_cano.sss_waterfall_image1,
                patch_cano.annotated_keypoints1,
                desc[0],
                canomax,
                kp_size=kp_size)
            patch_cano_annotated_kps2, patch_cano_desc2, patch_cano_img2_normalized = compute_desc_at_annotated_locations(
                patch_cano.sss_waterfall_image2,
                patch_cano.annotated_keypoints2,
                desc[0],
                canomax,
                kp_size=kp_size)
            patch_raw_annotated_kps1, patch_raw_desc1, patch_raw_img1_normalized = compute_desc_at_annotated_locations(
                patch_raw.sss_waterfall_image1,
                patch_raw.annotated_keypoints1,
                desc[0],
                rawmax,
                kp_size=kp_size)
            patch_raw_annotated_kps2, patch_raw_desc2, patch_raw_img2_normalized = compute_desc_at_annotated_locations(
                patch_raw.sss_waterfall_image2,
                patch_raw.annotated_keypoints2,
                desc[0],
                rawmax,
                kp_size=kp_size)
        else:
            patch_cano_annotated_kps1, patch_cano_desc1, patch_cano_img1_normalized = compute_desc_at_annotated_locations(
                patch_cano.sss_waterfall_image1,
                patch_cano.annotated_keypoints1,
                desc[0],
                canomax,
                kp_size=kp_size)
            rotated_cano_img2, rotated_cano_kps2 = patch_rotated(
                patch_cano.sss_waterfall_image2,
                patch_cano.annotated_keypoints2)
            patch_cano_annotated_kps2, patch_cano_desc2, patch_cano_img2_normalized = compute_desc_at_annotated_locations(
                rotated_cano_img2,
                rotated_cano_kps2,
                desc[0],
                canomax,
                kp_size=kp_size)

            patch_raw_annotated_kps1, patch_raw_desc1, patch_raw_img1_normalized = compute_desc_at_annotated_locations(
                patch_raw.sss_waterfall_image1,
                patch_raw.annotated_keypoints1,
                desc[0],
                rawmax,
                kp_size=kp_size)
            rotated_raw_img2, rotated_raw_kps2 = patch_rotated(
                patch_raw.sss_waterfall_image2, patch_raw.annotated_keypoints2)
            patch_raw_annotated_kps2, patch_raw_desc2, patch_raw_img2_normalized = compute_desc_at_annotated_locations(
                rotated_raw_img2,
                rotated_raw_kps2,
                desc[0],
                rawmax,
                kp_size=kp_size)

#       fig, (ax1, ax2) = plt.subplots(1, 2)
#       ax1.imshow(patch_raw_img1_normalized, cmap='gray')
#
#       ax1.scatter([a.pt[0] for a in patch_raw_annotated_kps1],
#                   [a.pt[1] for a in patch_raw_annotated_kps1],
#                   s=5,
#                   c='y')
#       ax2.imshow(patch_raw_img2_normalized, cmap='gray')
#       ax2.scatter([a.pt[0] for a in patch_raw_annotated_kps2],
#                   [a.pt[1] for a in patch_raw_annotated_kps2],
#                   s=5,
#                   c='y')
#       plt.savefig(
#           f'{patch_outpath}/{descType}-raw-fixed-bin-{fixed_size}-{patch_raw.filename1}-{patch_raw.filename2}-patch-{patch_raw.patch_id}.png'
#       )
#
#       fig, (ax1, ax2) = plt.subplots(1, 2)
#       ax1.imshow(patch_cano_img1_normalized, cmap='gray')
#       ax2.imshow(patch_cano_img2_normalized, cmap='gray')
#       ax1.scatter([a.pt[0] for a in patch_cano_annotated_kps1],
#                   [a.pt[1] for a in patch_cano_annotated_kps1],
#                   s=5,
#                   c='y')
#       ax2.scatter([a.pt[0] for a in patch_cano_annotated_kps2],
#                   [a.pt[1] for a in patch_cano_annotated_kps2],
#                   s=5,
#                   c='y')
#       plt.savefig(
#           f'{patch_outpath}/{descType}-cano-fixed-bin-{fixed_size}-{patch_cano.filename1}-{patch_cano.filename2}-patch-{patch_cano.patch_id}.png'
#       )
#       plt.close('all')

        raw_dist = np.linalg.norm(patch_raw_desc1 - patch_raw_desc2, axis=1)
        cano_dist = np.linalg.norm(patch_cano_desc1 - patch_cano_desc2, axis=1)
            
        for dist in raw_dist:
            file1 = patch_raw.filename1.split('_')[0]
            file2 = patch_raw.filename2.split('_')[0]
            row = pd.DataFrame({
                    'id': f'{file1}-{file2}-patch-{patch_raw.patch_id}',
                    'file1': file1,
                    'file2': file2,
                    'file_pair': f'{file1}-{file2}',
                    'patch_id': patch_raw.patch_id,
                    'dist': dist,
                    'is_raw': True,
                    'is_cano': False
                },
                index=[0])
            descriptors_df = pd.concat([descriptors_df,
                                        row]).reset_index(drop=True)
        for dist in cano_dist:
            file1 = patch_cano.filename1.split('_')[0]
            file2 = patch_cano.filename2.split('_')[0]
            row = pd.DataFrame({
                    'id': f'{file1}-{file2}-patch-{patch_cano.patch_id}',
                    'file1': file1,
                    'file2': file2,
                    'file_pair': f'{file1}-{file2}',
                    'patch_id': patch_cano.patch_id,
                    'dist': dist,
                    'is_raw': False,
                    'is_cano': True
                },
                index=[0])
            descriptors_df = pd.concat([descriptors_df,
                                        row]).reset_index(drop=True)


        # Chi-square
        cano_metric = similarity_compare(patch_cano_img1_normalized,
                                         patch_cano_img2_normalized)
        raw_metric = similarity_compare(patch_raw_img1_normalized,
                                        patch_raw_img2_normalized)
        imprv = (cano_metric - raw_metric) / raw_metric
        similarity_comparision.append([cano_metric, raw_metric, imprv])

        # KL divergence
        cano_metric = similarity_compare(patch_cano_img1_normalized,
                                         patch_cano_img2_normalized, kl_div=True)
        raw_metric = similarity_compare(patch_raw_img1_normalized,
                                        patch_raw_img2_normalized, kl_div=True)
        imprv = (cano_metric - raw_metric) / raw_metric
        similarity_comparision_kl.append([cano_metric, raw_metric, imprv])

        # Correlation
        cano_corr = correlation(patch_cano)
        raw_corr = correlation(patch_raw)
        corr_imprv = (cano_corr - raw_corr) / (raw_corr)
        corr_comparison.append([cano_corr, raw_corr, corr_imprv])

        if matcher == "Matcher":
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            patch_cano_correct = []
            patch_raw_correct = []
            patch_cano_matches = bf.match(patch_cano_desc1, patch_cano_desc2)
            patch_raw_matches = bf.match(patch_raw_desc1, patch_raw_desc2)
            for k in patch_cano_matches:
                if k.trainIdx == k.queryIdx:
                    patch_cano_correct.append(k)


#           cano_match_img = cv.drawMatches(
#               patch_cano_img1_normalized,
#               patch_cano_annotated_kps1,
#               patch_cano_img2_normalized,
#               patch_cano_annotated_kps2,
#               [x for x in patch_cano_matches if x not in patch_cano_correct],
#               None,
#               matchColor=[255, 0, 0])
#           cano_match_img_corr = cv.drawMatches(patch_cano_img1_normalized,
#                                                patch_cano_annotated_kps1,
#                                                patch_cano_img2_normalized,
#                                                patch_cano_annotated_kps2,
#                                                patch_cano_correct,
#                                                None,
#                                                matchColor=[0, 255, 0])
#           cano_match_img_gt = cv.drawMatches(
#               patch_cano_img1_normalized,
#               patch_cano_annotated_kps1,
#               patch_cano_img2_normalized,
#               patch_cano_annotated_kps2, [
#                   cv.DMatch(i, i, 0)
#                   for i in range(len(patch_cano_annotated_kps1))
#               ],
#               None,
#               matchColor=[0, 255, 0])
#           plt.imshow(
#               np.hstack(
#                   [cano_match_img, cano_match_img_corr, cano_match_img_gt]))
#           plt.show()

# if len(patch_cano_matches) > 0:
#     accuracy_patch_cano = len(patch_cano_correct) / len(patch_cano_matches)
#     print(f'Canonical accuracy, {accuracy_patch_cano}...... Correct, {len(patch_cano_correct)}...... Matched, {len(patch_cano_matches)}...... Patch no, {patch_comparision[i,0]}')

            for k in patch_raw_matches:
                if k.trainIdx == k.queryIdx:
                    patch_raw_correct.append(k)
            # if len(patch_raw_matches) > 0:
            #     accuracy_patch_raw = len(patch_raw_correct) / len(patch_raw_matches)
            #     print(f'Raw accuracy, {accuracy_patch_raw}...... Correct, {len(patch_raw_correct)}...... Matched, {len(patch_raw_matches)}...... Patch no, {patch_comparision[i,0]}')
            cano_correct.append(len(patch_cano_correct))
            cano_matched.append(len(patch_cano_matches))
            raw_correct.append(len(patch_raw_correct))
            raw_matched.append(len(patch_raw_matches))

        else:
            if descType == 'sift':
                distance_metric = cv.NORM_L2
            else:
                distance_metric = cv.NORM_HAMMING
            bf = cv.BFMatcher(distance_metric)
            ratio = desc[1]
            patch_cano_matches = bf.knnMatch(patch_cano_desc1,
                                             patch_cano_desc2,
                                             k=2)
            patch_raw_matches = bf.knnMatch(patch_raw_desc1,
                                            patch_raw_desc2,
                                            k=2)
            patch_cano_good = []
            patch_cano_correct = []
            patch_cano_incorrect = []
            patch_raw_good = []
            patch_raw_correct = []
            patch_raw_incorrect = []

            if len(patch_cano_matches) > 1:
                for m, n in patch_cano_matches:
                    if m.distance < ratio * n.distance:
                        patch_cano_good.append([m])
                for k in patch_cano_good:
                    if k[0].trainIdx == k[0].queryIdx:
                        patch_cano_correct.append(k[0])
                    else:
                        patch_cano_incorrect.append(k[0])
                    print(k[0].distance)
                # if len(patch_cano_good) > 0:
                #     accuracy_patch_cano = len(patch_cano_correct) / len(patch_cano_good)
                # print(f'Canonical accuracy, {accuracy_patch_cano}...... Correct, {len(patch_cano_correct)}...... Matched, {len(patch_cano_good)}...... Patch no, {patch_comparision[i,0]}')

            if len(patch_raw_matches) > 1:
                for m, n in patch_raw_matches:
                    if m.distance < ratio * n.distance:
                        patch_raw_good.append([m])
                for k in patch_raw_good:
                    if k[0].trainIdx == k[0].queryIdx:
                        patch_raw_correct.append(k[0])
                    else:
                        patch_raw_incorrect.append(k[0])
                # if len(patch_raw_good) > 0:
                #     accuracy_patch_raw = len(patch_raw_correct) / len(patch_raw_good)
                #     print(f'Raw accuracy, {accuracy_patch_raw}...... Correct, {len(patch_raw_correct)}...... Matched, {len(patch_raw_good)}...... Patch no, {patch_comparision[i,0]}')

            # patch_cano_matched_img = cv.drawMatchesKnn(patch_cano_img1_normalized,patch_cano_annotated_kps1,patch_cano_img2_normalized,patch_cano_annotated_kps2,patch_cano_good, None, flags=2)
            # # patch_cano_matched_img = cv.drawMatches(patch_cano_img1_normalized,patch_cano_annotated_kps1,patch_cano_img2_normalized,patch_cano_annotated_kps2,patch_cano_matches, None, flags=2)
            # plt.figure()
            # plt.title('canonical' + patch_comparision[i,0])
            # plt.imshow(patch_cano_matched_img)

            # patch_raw_matched_img = cv.drawMatchesKnn(patch_raw_img1_normalized,patch_raw_annotated_kps1,patch_raw_img2_normalized,patch_raw_annotated_kps2,patch_raw_good, None, flags=2)
            # # patch_raw_matched_img = cv.drawMatches(patch_raw_img1_normalized,patch_raw_annotated_kps1,patch_raw_img2_normalized,patch_raw_annotated_kps2,patch_raw_matches, None, flags=2)
            # plt.figure()
            # plt.imshow(patch_raw_matched_img)
            # plt.title('raw' + patch_comparision[i,0])

            cano_correct.append(len(patch_cano_correct))
            cano_matched.append(len(patch_cano_good))
            raw_correct.append(len(patch_raw_correct))
            raw_matched.append(len(patch_raw_good))

            plot_matching_results(img1=patch_raw_img1_normalized,
                                  kps1=patch_raw_annotated_kps1,
                                  filename1=patch_raw.filename1,
                                  img2=patch_raw_img2_normalized,
                                  kps2=patch_raw_annotated_kps2,
                                  filename2=patch_raw.filename2,
                                  matches_correct=patch_raw_correct,
                                  matches_incorrect=patch_raw_incorrect,
                                  patch_id=patch_raw.patch_id,
                                  filetype='raw',
                                  patch_outpath=patch_outpath,
                                  descType=descType)

            plot_matching_results(img1=patch_cano_img1_normalized,
                                  kps1=patch_cano_annotated_kps1,
                                  filename1=patch_cano.filename1,
                                  img2=patch_cano_img2_normalized,
                                  kps2=patch_cano_annotated_kps2,
                                  filename2=patch_cano.filename2,
                                  matches_correct=patch_cano_correct,
                                  matches_incorrect=patch_cano_incorrect,
                                  patch_id=patch_cano.patch_id,
                                  filetype='cano',
                                  patch_outpath=patch_outpath,
                                  descType=descType)

    # print(f'Raw correct, {sum(raw_correct)}...... Raw matched, {sum(raw_matched)}')
    # print(f'Cano correct, {sum(cano_correct)}...... Cano matched, {sum(cano_matched)}')
    # print('#################################')
    path_sep = patch_outpath.split(os.sep)
    filename_pair = path_sep[-3] + '-' + path_sep[-1]
    cano_match.append(
        [sum(cano_correct),
         sum(cano_matched), descType, filename_pair])
    raw_match.append(
        [sum(raw_correct),
         sum(raw_matched), descType, filename_pair])

    return cano_match, raw_match, similarity_comparision, similarity_comparision_kl, corr_comparison, descriptors_df


def similarity_compare(patch_img1_normalized, patch_img2_normalized, kl_div=False):
    hist2 = cv.calcHist(patch_img2_normalized, [0], None, [256], [0, 255])
    hist1 = cv.calcHist(patch_img1_normalized, [0], None, [256], [0, 255])
    # plt.figure()
    # plt.hist(hist1)
    # plt.figure()
    # plt.hist(hist2)
    # plt.show()
    #print(len(hist1))
    if kl_div:
        similarity = cv.compareHist(hist1, hist2, cv.HISTCMP_KL_DIV)
    else:
        similarity = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR_ALT)
    return similarity

def correlation(patch):
    img1 = patch.sss_waterfall_image1
    img2 = patch.sss_waterfall_image2
    if img1.shape != img2.shape:
        print(f'Reshaping patch: ({patch.filename1}, {patch.filename2}) where img1.shape = {img1.shape}, img2.shape = {img2.shape}')

        min_bins = min(img1.shape[1], img2.shape[1])
        img1 = img1[:, :min_bins]
        img2 = img2[:, :min_bins]
        
    return (np.multiply(img1, img2)).sum() / np.sqrt((np.square(img1)).sum()) / np.sqrt((np.square(img2)).sum())

def plot_matching_results(img1, kps1, filename1,
                          img2, kps2, filename2,
                          matches_correct, matches_incorrect,
                          patch_id, filetype,
                          patch_outpath,
                          descType):
    offset = img1.shape[1] + 1

    # Plot ground truth matches
    _, ax_gt = plt.subplots()
    _plot_image_pairs(ax_gt, img1, img2)
    plt.axis('off')
    for kp_in_img1, kp_in_img2 in zip(kps1, kps2):
        x = [kp_in_img1.pt[0], kp_in_img2.pt[0]+offset]
        y = [kp_in_img1.pt[1], kp_in_img2.pt[1]]
        plt.plot(x, y, '-o', color='#00ff26', linewidth=1, markersize=2)
    plt.savefig(f'{patch_outpath}/{filename1}-{filename2}-patch{patch_id}-{filetype}-gt.png',
            dpi=200,
            bbox_inches='tight',
            pad_inches=0)

    # Plot matches
    _, ax_matches = plt.subplots()
    _plot_image_pairs(ax_matches, img1, img2)
    plt.axis('off')
    for m in matches_correct:
        x = [kps1[m.queryIdx].pt[0], kps2[m.trainIdx].pt[0]+offset]
        y = [kps1[m.queryIdx].pt[1], kps2[m.trainIdx].pt[1]]
        plt.plot(x, y, '-o', color='#00ff26', linewidth=1, markersize=2)

    for m in matches_incorrect:
        x = [kps1[m.queryIdx].pt[0], kps2[m.trainIdx].pt[0]+offset]
        y = [kps1[m.queryIdx].pt[1], kps2[m.trainIdx].pt[1]]
        plt.plot(x, y, '-o', color='r', linewidth=1, markersize=2)
    plt.savefig(f'{patch_outpath}/{filename1}-{filename2}-patch{patch_id}-{filetype}-results-{descType}.png',
            dpi=200,
            bbox_inches='tight',
            pad_inches=0)
    plt.close('all')


def _plot_image_pairs(ax, img1, img2):
    ax.imshow(np.hstack([img1, np.zeros((img1.shape[0], 1)), img2]), cmap='gray')
    ax.vlines(x=img1.shape[1]+1, ymin=0, ymax=img1.shape[0]-1, colors='white')
    return ax
