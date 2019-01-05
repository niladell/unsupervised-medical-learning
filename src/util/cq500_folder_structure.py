import os
import json

# path = '/home/pereira_inez_gmail_com/bucket_tmp/CT_head_trauma_unzipped'
path = os.getcwd()

os.chdir(path)
print(os.getcwd())

def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

list = get_immediate_subdirectories(path)

list2 = []
for dir in list:
    if not dir.startswith('.'):
        list2.append(dir)

count = 0
dic_of_filenames = dict()
dic_folder_to_subjects = dict()
for subject in list2:
    # print(subject)
    if os.listdir(os.path.join(path, subject)) == ['Unknown Study']:
        # print(subject)
        # print(os.listdir(os.path.join(path, subject)))
        count +=1
        list_subdir = get_immediate_subdirectories(os.path.join(path, subject, 'Unknown Study'))
        for subdir in list_subdir:
            # print(subdir)
            if subdir not in dic_of_filenames.keys():
                dic = {subdir: 1}
                # print(subject, subdir)
                dic_of_filenames.update(dic)
                dic2 = {subdir: [subject]}
                dic_folder_to_subjects.update(dic2)
            else:
                dic_of_filenames[subdir] +=1
                dic_folder_to_subjects[subdir].append(subject)
    else:
        print("I can't find that 'Unknown Study' folder for "+subject)

dic_of_filenames
dic_folder_to_subjects

with open("_subfolder_frequency.json", "w") as fp:
    json.dump(dic_of_filenames, fp)

with open("_subfolder_per_subject.json", "w") as fp:
    json.dump(dic_folder_to_subjects, fp)