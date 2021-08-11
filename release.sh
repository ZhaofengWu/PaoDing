#!/bin/bash

set -Exeuo pipefail

if [[ ${#} -ne 1 ]]; then
    echo "usage: bash release.sh VERSION"
    exit 1
fi

version=$1
release_folder=release-${version}

if [[ -d ${release_folder} ]]; then
    echo "${release_folder} exists"
    exit 1
fi

git tag ${version} -m "Adds tag ${version} for pypi"
git push --tags origin master -f

# Fresh copy
git clone $(git remote get-url origin) ${release_folder}
cd ${release_folder}
git checkout ${version}

python -m build
twine upload dist/*

rm -rf ${release_folder}
