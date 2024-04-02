#!/bin/bash

sampleList=/home/chenfy/projects/seg/data/meta/selected_seg_meta_index.txt

while read sample
do

outfile=/home/chenfy/projects/seg/images/largebone2d/$sample.jpg
if [ -f "$outfile" ]
then
	echo "$sample croppings already exist"
else
/data6/cfy/anaconda3/envs/pytorch/bin/python /home/chenfy/projects/seg/codes/select_bone.py $sample
fi

done < $sampleList