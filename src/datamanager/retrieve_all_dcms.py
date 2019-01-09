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
# test_string = '/Users/ines/Desktop/subjects copy 2/subject108/Unknown Study/CT 0.625mm/CT000253.dcm'
# re_forbidden_folders.search(test_string)

def get_dcms(path):
    list_of_dcm = []
    for dirpath, dirname, filenames in os.walk(path):
        # print(filenames)
        # print(dirpath)
        for file in filenames:
            pattern = re.compile(r'.dcm$')
            m = re.search(pattern, file)
            if m is not None and re_forbidden_folders.search(dirpath) is None:
                dcm_path = dirpath + '/' + file
                list_of_dcm.append(dcm_path)
    return list_of_dcm

# Testing results:
# for path in list_of_dcm:
#     re_forbidden_folders.search(path)

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    # print(dirname)

    path = '/Users/ines/Desktop/subjects copy 2'
    list_of_files = get_dcms(path)

    os.chdir(dirname)
    with open('list_of_dcms.txt', 'w') as f:
        for item in list_of_files:
            f.write("%s\n" % item)
