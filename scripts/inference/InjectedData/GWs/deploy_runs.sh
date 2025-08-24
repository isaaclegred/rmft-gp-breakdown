for INI_FILE in $(ls *.ini)
do
    lwp-pipe $INI_FILE
done
