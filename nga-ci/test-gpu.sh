#!/usr/bin/env bash

set -x
set -e

. /opt/spack/share/spack/setup-env.sh
spack env activate trilinos

pushd /opt/build/Trilinos
ret_code=0
ctest -j 5 --output-on-failure || ret_code=$?
# We collect the test logs for exporting
echo "ctest returned: $ret_code"
mkdir -p /tmp/artifacts/
cp /opt/build/Trilinos/Testing/Temporary/LastTest.log /tmp/artifacts/
echo ${ret_code} > /tmp/artifacts/success_flag.txt
ls /tmp/artifacts
popd
