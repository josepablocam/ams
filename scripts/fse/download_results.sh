#!/usr/bin/env bash
source scripts/folder_setup.sh

obj_name="fse-artifact-results"

wget "https://ams-fse.s3.us-east-2.amazonaws.com/${obj_name}.zip"
unzip "${obj_name}.zip"
echo "Moving files to ${RESULTS} and ${ANALYSIS_DIR}"
mv "${obj_name}/results" ${RESULTS}
mv "${obj_name}/analysis_results" ${ANALYSIS_DIR}
rm -rf ${obj_name}
# but prompt before removing the actual zip file
rm -i "${obj_name}.zip"
