#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a Serial Multi-Process job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J prod_heatrate

#SBATCH -A acnntopo

# Standard out and Standard Error output files
#SBATCH -o log%A.output
#SBATCH -e log%A.error

#To send emails, set the adcdress below and remove one of the "#" signs.
##SBATCH --mail-user aae8800@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
##SBATCH -t 5-0:0:0

#SBATCH -p tier3 -n 20
##SBATCH -p debug -n 6
##SBATCH --exclude=skl-a-03

# Job memory requirements in MB
#SBATCH --mem-per-cpu=8000

module load openmpi
module load gcc

input_parameters="BaseAcidRatio AshContent NaContent FeContent BTU H2OContent SContent AlContent CaContent KContent MgContent SiContent SOContent TiContent Cold_Reheat_Steam_Temperature ECON_FLUE_GAS_OUT_PRESS ECON_HDR_01_OUT_TEMP ECON_HDR_02_OUT_TEMP ECON_IN_GAS_TEMP ECONOMIZER_DIFFERENTIAL ECONOMIZER_GAS_OUTLET_O2_LEVEL ECONOMIZER_INLET_FEEDWATER_FLOW ECONOMIZER_INLET_FEEDWATER_TEMPERATURE ECONOMIZER_OUTLET_AVG GROSS_GENERATOR_OUTPUT HOT_REHEAT_TEMPERATURE_REHEATER_OUTLET MAIN_STEAM_PRESSURE_AT_BOILER MAIN_STEAM_TEMP_SUPERHEATER_OUTLET Main_Steam_Press Main_Steam_Spray_Flow Main_Steam_Spray_Press NOX Net_Plant_Heat_Rate Net_Unit_Generation PRIM_SUPHTR_DIFFERENTIAL PSH_Gas_Outlet_Temperature PSH_OUTLET_AVG RH_SUPHTR_BANK_1_DIFF RH_SUPHTR_BANK_2_DIFF SSH_INLET_HDR_01_TEMP SSH_INLET_HDR_04_TEMP SSH_OUT_HDR_TC_01_TEMP SSH_OUT_HDR_TC_02_TEMP SSH_OUT_HDR_TC_03_TEMP SSH_OUT_HDR_TC_04_TEMP SSH_OUTLET_HDR_TC_05_TEMP OFA_Flow"
output_parameter="Net_Plant_Heat_Rate"

max_gen=20000; 
population=20; 
island=10;


process () {
		#for folder in  0 1 2 3 4 5; do
		for folder in  0 1 2 3 4 5 6 7 8 9 10 11; do
			
			exp_name="./heatrate_preditions/$offset/$folder"
			mkdir -p $exp_name
			file=$exp_name"/fitness_log.csv"
			time srun ./exact/build/mpi/colony --training_filenames data_heatrate_nose/train_.csv --test_filenames data_heatrate_nose/test_.csv --time_offset $offset --input_parameter_names $input_parameters --output_parameter_names $output_parameter --population_size $population --max_genomes $max_gen --bp_iterations 40 --normalize min_max --output_directory $exp_name --max_recurrent_depth 10 --number_islands $island --std_message_level ERROR --file_message_level ERROR
		echo "############**********###########**********###########"
		done
}

for offset in 1 2 4 8; do
	process
done


:<< 'END'

END
