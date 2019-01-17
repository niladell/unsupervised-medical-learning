import os
import csv
import random

def list_creator(relative_path):
    with open(relative_path, 'r') as f:
        list_of_subjects = f.read().splitlines()
    return list_of_subjects

healthy = list_creator('scripts/healthy.csv')


selected_healthy = random.sample(healthy, 20)

selected_healthy_corrected = []
for subject in selected_healthy:
    selected_healthy_corrected.append(subject.replace('-', ''))

os.chdir('/Users/ines/Dropbox/CT_head_trauma')

for file in os.listdir():
    if file.startswith()