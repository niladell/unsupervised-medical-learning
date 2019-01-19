import os
import random
import re
from scripts.retrieve_all_dcms import get_dcms


def list_creator(relative_path):
    with open(relative_path, 'r') as f:
        list_of_subjects = f.read().splitlines()
    return list_of_subjects

def correct_hyphenation(selected_subjects):
    selected_subjects_corrected = []
    for subject in selected_subjects:
        selected_subjects_corrected.append(subject.replace('-', ''))
    return selected_subjects_corrected


def get_file_paths(re_selected):
    dcms = []
    for file in os.listdir():
        m = re_selected.search(file)
        if m is not None:
            dcms.extend(get_dcms(file))
    return dcms

def main(relative_path, number, output_name):
    list_of_subjects = list_creator(relative_path)
    selected = random.sample(list_of_subjects, number)
    selected_corrected = correct_hyphenation(selected)
    re_selected = re.compile(r'\b(?:%s)\b' % '|'.join(selected_corrected))
    dcms = get_file_paths(re_selected)

    with open(output_name, 'w') as f:
        for item in dcms:
            f.write("%s\n" % item)

    return dcms


if __name__ == '__main__':

    healthy = main('healthy.csv', 20, 'healthy.txt')
    hemorrhage = main('hemorrhages.csv', 20, 'hemorrhage.txt')
    fractures = main('fractures.csv', 20, 'fractures.txt')


# ###########
# # HEALTHY #
# ###########
#
# healthy = list_creator('scripts/healthy.csv')
# selected_healthy = random.sample(healthy, 20)
# # Already preprocessed and put here:
# selected_healthy_corrected =    ['CQ500CT53',
#                                 'CQ500CT88',
#                                 'CQ500CT294',
#                                 'CQ500CT70',
#                                 'CQ500CT463',
#                                 'CQ500CT395',
#                                 'CQ500CT98',
#                                 'CQ500CT482',
#                                 'CQ500CT91',
#                                 'CQ500CT117',
#                                 'CQ500CT472',
#                                 'CQ500CT387',
#                                 'CQ500CT25',
#                                 'CQ500CT152',
#                                 'CQ500CT306',
#                                 'CQ500CT124',
#                                 'CQ500CT103',
#                                 'CQ500CT471',
#                                 'CQ500CT89',
#                                 'CQ500CT201']
#
#
# # selected_healthy_corrected = ['CQ500CT18', 'CQ500CT0', 'CQ500CT17']
#
# re_selected_healthy = re.compile(r'\b(?:%s)\b' % '|'.join(selected_healthy_corrected))
#
# # os.chdir('/Users/ines/Dropbox/CT_head_trauma')
#
# with open('healthy_dcms.txt', 'w') as f:
#     for item in dcms:
#         f.write("%s\n" % item)
#
# ##############
# # HEMORRHAGE #
# ##############
#
# hemorrhage = list_creator('scripts/hemorrhages.csv')
# selected_hemorrhage = random.sample(hemorrhage, 20)
# selected_hemorrhage_corrected = correct_hyphenation(selected_hemorrhage)
#
# # These functions yield:
# selected_hemorrhage_corrected =  ['CQ500CT53',
#                                  'CQ500CT88',
#                                  'CQ500CT294',
#                                  'CQ500CT70',
#                                  'CQ500CT463',
#                                  'CQ500CT395',
#                                  'CQ500CT98',
#                                  'CQ500CT482',
#                                  'CQ500CT91',
#                                  'CQ500CT117',
#                                  'CQ500CT472',
#                                  'CQ500CT387',
#                                  'CQ500CT25',
#                                  'CQ500CT152',
#                                  'CQ500CT306',
#                                  'CQ500CT124',
#                                  'CQ500CT103',
#                                  'CQ500CT471',
#                                  'CQ500CT89',
#                                  'CQ500CT201']
#
# re_selected_hemorrhage = re.compile(r'\b(?:%s)\b' % '|'.join(selected_hemorrhage_corrected))
# dcms = get_file_paths(re_selected_hemorrhage)
#
#

