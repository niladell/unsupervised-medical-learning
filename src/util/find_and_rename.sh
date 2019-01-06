#!/bin/bash
# run from iowa_bucket/CT_head_trauma_unzipped

# Folder structure initially 

#cq500_sample
#	CQ500CT0 CQ500CT0
#		Unknown Study
#			CT 4cc sec 150cc D3D on-2
#				CT000000.dcm
#				CT000001.dcm
#				CT000002.dcm
#				...
#
#Ziel: 000_cc-sec-150cc-D3D-on-2_CT000001.dcm


# replace all whitespaces with underscore in ALL subfolders
find . -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;

# rename 1: remove "double name": CQ500CT0 CQ500CT0
rename 's\_.*\\' *

# loop through all subfolders and remove "Unknown_Study layer"
for d in *
do
    ( cd "$d/Unknown_Study" && find . -maxdepth 1 -exec mv {} .. \; && rm -r ../"Unknown_Study")
done

# rename 2: name every picture after its parent directories (replace cq500_sample_test!)
for dir in *; do 
    find "$dir" -type f -print0 | 
        while IFS= read -r -d '' f; do 
            dd=$(dirname "$f")
            new="${f/CT_head_trauma_unzipped\/}"
            new="${new//\//_}" 
            mv "$f" "$dd"/"$new"
        done
done

# move all dcm files to parent folder
find . -mindepth 2 -type f -print -exec mv {} . \;

# remove all empty folders
find . -type d -empty -delete

# exclude all files ines discarded
find . -type f -name '*.CT_4cc_sec_150cc_D3D_on.*' -delete
find . -type f -name '*.CT_POST_CONTRAST.*' -delete
find . -type f -name '*.CT_I_To_S.*' -delete
find . -type f -name '*.CT_PRE_CONTRAST_BONE.*' -delete
find . -type f -name '*.CT_Thin_Bone.*' -delete
find . -type f -name '*.CT_Thin_Stnd.*' -delete
find . -type f -name '*.CT_0.625mm.*' -delete
find . -type f -name '*.CT_Thin_Details.*' -delete
# remove all gstmp files
find . -type f -name '*.gstmp' -delete
