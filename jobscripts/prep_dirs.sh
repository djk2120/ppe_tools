# set up some directory stuff, here and in scratch

codebase='PPEn11'
envtypes=('CTL2010' 'C285' 'AF1855')
runtypes=('hist_AD' 'hist_SASU' 'hist_postSASU' 'hist')



basedir="/glade/scratch/djk2120/"


for envtype in ${envtypes[@]}; do
    for runtype in ${runtypes[@]}; do
	mkdir -p $basedir$codebase"/"$runtype"/"$envtype
    done
done

mkdir -p $basedir$codebase"/restarts"
mkdir -p $basedir$codebase"/namelist_mods"
mkdir -p $basedir$codebase"/paramfiles"

mkdir -p $codebase"/configs"
mkdir -p $codebase"/nlbase"
