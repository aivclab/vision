#!/usr/bin/env bash

PARENT="$( dirname -- "$0"; )";
cd "${PARENT}" || exit

rm -rf github
rm -rf source/generated
