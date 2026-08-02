#!/bin/bash

# Define the exact files in an array
CSV_FILES=(
  "csv_timeseries_pyelastica/case_1_fixed_10mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_2_fixed_50mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_3_random_1_10mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_4_random_10_80mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_5_ramp_1_10mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_6_ramp_10_80mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_7_micro_0.1_1mm/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_8_free_relaxation/collected_trajectories_koopman_timeseries.csv"
  "csv_timeseries_pyelastica/case_9_long_mixed/collected_trajectories_koopman_timeseries.csv"
)

# Run the python script
python3 neural_koopman.py --csv \
  "${CSV_FILES[4]}" \
  "${CSV_FILES[5]}" \
  "${CSV_FILES[6]}" \
  "${CSV_FILES[7]}" \
  "${CSV_FILES[8]}" \
  --horizon 10