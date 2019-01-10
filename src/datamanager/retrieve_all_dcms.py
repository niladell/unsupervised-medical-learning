import os
import re

list_forbidden_folders = ['CT 4cc sec 150cc D3D on',
                          'CT 4cc sec 150cc D3D on-2',
                          'CT 4cc sec 150cc D3D on-3',
                          'CT POST CONTRAST',
                          'CT POST CONTRAST-2',
                          'CT BONE',
                          'CT I To S',
                          'CT PRE CONTRAST BONE',
                          'CT Thin Bone',
                          'CT Thin Stnd',
                          'CT 0.625mm',
                          'CT 0.625mm-2',
                          'CT 5mm POST CONTRAST',
                          'CT ORAL IV',
                          'CT 55mm Contrast',
                          'CT BONE THIN',
                          'CT 3.753.75mm Plain',
                          'CT Thin Details',
                          'CT Thin Stand']

re_forbidden_folders = re.compile(r'\b(?:%s)\b' % '|'.join(list_forbidden_folders))

def get_dcms(path):
    list_of_dcm = []
    for dirpath, dirname, filenames in os.walk(path):
        for file in filenames:
            pattern = re.compile(r'.dcm$')
            m = re.search(pattern, file)
            if m is not None and re_forbidden_folders.search(dirpath) is None:
                dcm_path = dirpath + '/' + file
                list_of_dcm.append(dcm_path)
    return list_of_dcm

if __name__ == '__main__':
    path = '.'
    list_of_files = get_dcms(path)

    with open('list_of_dcms.txt', 'w') as f:
        for item in list_of_files:
            f.write("%s\n" % item)
