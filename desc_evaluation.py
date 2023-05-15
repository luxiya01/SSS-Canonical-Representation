import sys

sys.path.append('/home/li/Documents/auvlib/scripts/')
#sys.path.append('/home/viki/Master_Thesis/auvlib/scripts')
#sys.path.append('/home/viki/Master_Thesis/SSS-Canonical-Representation')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, patch_rotated
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from canonical_transformation import canonical_trans
from desc_gen_script import desc_evaluation, similarity_compare
import cv2 as cv
from tabulate import tabulate
import pandas as pd

#data_path = './data/ssh'
#data_path = './SSH_Pairs/ssh'
#model = 'sin_square'
#descType = 'sift'
#fixed_size = False
data_path = sys.argv[1]
parent_folder = '/'.join(data_path.split('/')[:-1])
model = sys.argv[2]
descType = sys.argv[3]
fixed_size = bool(int(sys.argv[4]))
print(model, descType, fixed_size)

sss_no = ['170', '171', '172', '173', '174']
cano_match = []
raw_match = []
similarity_comparision = []
similarity_comparision_kl = []
corr_comparison = []
descriptors = {
    'raw': {
        'img1': [],
        'img2': []
    },
    'cano': {
        'img1': [],
        'img2': []
    }
}
descriptors_df = pd.DataFrame(
    columns=['id', 'file1', 'file2', 'file_pair', 'patch_id', 'dist', 'is_raw', 'is_cano'])

for i in sss_no:
    for j in sss_no:
        if i == j:
            continue
        #patch_outpath = data_path + i + '/patch_pairs/ssh' + j
        patch_outpath = data_path + i + f'/{model}_patch_pairs/ssh' + j
        matcher = 'knnMatcher'
        rotate = False
        if (int(i) + int(j)) % 2:
            rotate = True
        cano_match, raw_match, similarity_comparision, similarity_comparision_kl, corr_comparison, descriptors_df = desc_evaluation(
            patch_outpath, matcher, descType, rotate, cano_match, raw_match,
            similarity_comparision,
            similarity_comparision_kl,
            corr_comparison, descriptors_df,
            fixed_size=fixed_size)
#        descriptors['raw']['img1'].extend(patch_descriptors['raw']['img1'])
#        descriptors['raw']['img2'].extend(patch_descriptors['raw']['img2'])
#        descriptors['cano']['img1'].extend(patch_descriptors['cano']['img1'])
#        descriptors['cano']['img2'].extend(patch_descriptors['cano']['img2'])

cano_match = np.array(cano_match, dtype=object)
raw_match = np.array(raw_match, dtype=object)

similarity_comparision = np.array(similarity_comparision)
imprv_res = similarity_comparision[:, 2]

similarity_comparision_kl = np.array(similarity_comparision_kl)
imprv_res_kl = similarity_comparision_kl[:, 2]

corr_comparison = np.array(corr_comparison)
corr_imprv_res = corr_comparison[:, 2]

raw_correct_sum = raw_match[:, 0].sum()
raw_match_sum = raw_match[:, 1].sum()
cano_correct_sum = cano_match[:, 0].sum()
cano_match_sum = cano_match[:, 1].sum()
#print(tabulate(similarity_comparision))
print(
    f'Overall Raw correct, {raw_correct_sum}...... Raw matched, {raw_match_sum}...... ACC, {raw_correct_sum/raw_match_sum}'
)
print(
    f'Overall Cano correct, {cano_correct_sum}...... Cano matched, {cano_match_sum}...... ACC, {cano_correct_sum/cano_match_sum}'
)
print(
    f'Chi-Square distance of {len(imprv_res[imprv_res<0])} pairs out of {len(imprv_res)} pairs decrease, ratio: {len(imprv_res[imprv_res<0])/len(imprv_res)}\n'
    f'\t average score: raw = {similarity_comparision[:, 1].mean()}; cano = {similarity_comparision[:, 0].mean()}'
)
print(
    f'KL-divergence of {len(imprv_res_kl[imprv_res_kl<0])} pairs out of {len(imprv_res_kl)} pairs decrease, ratio: {len(imprv_res_kl[imprv_res_kl<0])/len(imprv_res_kl)}\n'
    f'\t average score: raw = {similarity_comparision_kl[:, 1].mean()}; cano = {similarity_comparision_kl[:, 0].mean()}'
)
print(
    f'Correlation of {len(corr_imprv_res[corr_imprv_res>0])} pairs out of {len(corr_imprv_res)} pairs increased, ratio: {len(corr_imprv_res[corr_imprv_res>0])/len(corr_imprv_res)}\n'
    f'\t average score: raw = {corr_comparison[:, 1].mean()}; cano = {corr_comparison[:, 0].mean()}'
)

#plt.hist(similarity_comparision[:, 2])
#plt.show()
print(tabulate(cano_match))
print(tabulate(raw_match))

# Compute descriptors distance
#descriptors['raw']['img1'] = np.concatenate(descriptors['raw']['img1'])
#descriptors['raw']['img2'] = np.concatenate(descriptors['raw']['img2'])
#descriptors['cano']['img1'] = np.concatenate(descriptors['cano']['img1'])
#descriptors['cano']['img2'] = np.concatenate(descriptors['cano']['img2'])
descriptors_df.to_csv(f'{parent_folder}/{descType}-{model}-{matcher}-fixed-bin-size-{fixed_size}.csv')
