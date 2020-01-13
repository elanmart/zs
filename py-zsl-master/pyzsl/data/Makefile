ifndef ROOT
    $(error Set ROOT first - directory where you want to keep your AWA dataset)
endif

all: AWA

#TODO(kaczorrrro) figure out a better way to set PYTHONPATH

AWA:
	@ PYTHONPATH='.' luigi --module awa.tasks RunAllTask --path ${ROOT} --local-schedule
	@ echo "Running validation"
	@ PYTHONPATH='.' luigi --module awa.tasks Validate --path ${ROOT} --local-schedule

# Remove all results of preprocessing
AWA_reset:
	@ rm -rf $(ROOT)/descriptions/
	@ rm -rf $(ROOT)/features/
	@ rm -rf $(ROOT)/indices/
	@ rm -rf $(ROOT)/labels/
	@ rm -rf $(ROOT)/metadata/
# Remove only unzipped stuff
AWA_clean_tmp:
	@ rm -f $(ROOT)/tmp
# Remove downloaded files
AWA_clean_downloads:
	@ rm -rf $(ROOT)/raw

AWA_clean_all: AWA_reset AWA_clean_tmp AWA_clean_downloads

clean: AWA_clean_all

# I hope that no one will ever call any file in this dir AWA_rest_of_task_name
.PHONY: all AWA clean

