#!/usr/bin/env bash

set -x
set -e

. /opt/spack/share/spack/setup-env.sh
spack env activate trilinos

cd /opt/build/Trilinos
ret_code=0

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# path to the artifacts
artifacts_dir=/tmp/artifacts

ctest -j 14 --output-junit junit-tests-report.xml --output-on-failure || ret_code=$?
# We collect the test logs for exporting
echo "ctest returned: $ret_code"
mkdir -p ${artifacts_dir}
cp /opt/build/Trilinos/junit-tests-report.xml ${artifacts_dir}
cp /opt/build/Trilinos/Testing/Temporary/LastTest.log ${artifacts_dir}
echo ${ret_code} > ${artifacts_dir}/success_flag.txt
ls ${artifacts_dir}
exit ${ret_code}
