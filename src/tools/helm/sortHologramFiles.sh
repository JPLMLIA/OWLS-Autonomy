#!/bin/bash
#=============================================================================
# @file sortHologramFiles.sh 
# @brief  Move .tif files of source directory and placed them in sub directories of specified number of files per session.
# @description
#            The command line arguments: 
#               -d the name of the bottom directory that contains the sequence of files in the session [Holograms]
#				-f the file suffix (e.g. tif, raw, bmp, etc) [tif]
#               -n the maximum number of files to put in session directory [600] 
#
#            'session_dir' must exist with subdirectory .<suffix> within it.  Original Files are left in the <seq_dir>
#            and placed into directories of the following structure.  
#
#            Also files 'nodemaps.txt' and 'timestamps.txt' are copied from 'session_dir' and placed into each new session directory.
#
#                <session_dir>/sess_000/<seq_dir>
#                <session_dir>/sess_001/<seq_dir>
#                <session_dir>/sess_002/<seq_dir>
#
#            Each sessXX directory will have at most 'num_files_per_session' sequential files.
#
#
# @author S. Felipe Fregoso, JPL <sfregoso@jpl.nasa.gov>
# @author C. Lindensmith, JPL <lindensm@jpl.nasa.gov>
# @date   Aug 5, 2022
# @rev    1.1
# 
#
#
#=============================================================================

usage() {
#    echo ""
#    echo "usage: sortHologramFiles.sh session_dir num_files_per_session"
#    echo ""
	echo "usage sortHologamFiles.sh -d [sequence directory name] -f [file suffix] -n [number] [path to parent directory of sequence directory]"
	echo "defaults:"
	echo "      -d Holograms"
	echo "      -f tif"
	echo "      -n 600"
	echo "the path to the parent directory accepts globs"
}

shopt -s nullglob


#set default values
SEQ_DIR="Holograms";
SUFFIX="tif";
FILES_PER_SESS=600;


while getopts d:f:n: flag
do
	case "${flag}" in
		d) SEQ_DIR=${OPTARG};;
		f) SUFFIX=${OPTARG};;
		n) FILES_PER_SESS=${OPTARG};;
	esac
done

shift $((OPTIND - 1)) #shift away any options
BASE_PATH="$@" #Directory path 




#FILES_PER_SESS=$2 # Maximum number of images per session directory
MIN_FILES_PER_SESS=10
MAX_FILES_PER_SESS=20000


echo "Sequence Directory: $SEQ_DIR"
echo "Image file suffix: $SUFFIX"
echo "Files per session: $FILES_PER_SESS"



# Range check FILES_PER_SESS
if [[ $FILES_PER_SESS -lt $MIN_FILES_PER_SESS || $FILES_PER_SESS -gt $MAX_FILES_PER_SESS ]]; then
    printf "ERROR.  Maximum number of files pers session [%d] must be between [%d] and [%d] inclusive." $FILES_PER_SESS $MIN_FILES_PER_SESS $MAX_FILES_PER_SESS
    exit
fi

# now loop over the sets of files

for dir in $BASE_PATH
do
	RAW_DIR_PATH="$dir/$SEQ_DIR"	#construct the directory path
	echo "Splitting Session: $RAW_DIR_PATH"

	# Copy over 'nodemaps.txt' and 'timestamps.txt'

	# Get a list of .tif files in the 'RAW_DIR_PATH' directory
	count=0
	for entry in "$RAW_DIR_PATH"/*".$SUFFIX"
	do

		new_sess_idx=$(($count/$FILES_PER_SESS))

		new_sess_dir=$(printf "%s/sess_%03d" $dir $new_sess_idx)
		if [[ ! -d $new_sess_dir ]]; then

			mkdir $new_sess_dir
			mkdir "$new_sess_dir/$SEQ_DIR"

			# Symlink strategy
			#ln -s $dir/nodemaps.txt "$new_sess_dir/nodemaps.txt"
			#ln -s $dir/timestamps.txt "$new_sess_dir/timestamps.txt"
		
			# Copy strategy
			cp -P $dir/nodemaps.txt $new_sess_dir/.
			cp -P $dir/timestamps.txt $new_sess_dir/.
		fi

		# Symlink strategy
		#fname=$(basename $entry)
		#ln -s $entry "$new_sess_dir/$SEQ_DIR/$fname"
	
		# Copy strategy
		cp $entry "$new_sess_dir/$SEQ_DIR"

		let "count+=1"
		#echo $entry
	done

	echo "Found $count sequenced files."
done