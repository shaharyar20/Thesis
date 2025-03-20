#!/bin/bash

black="black"

# Applies python-format

# check that we are in a clean state in order to prevent accidental changes
if [ ! -z "$(git status --untracked-files=no  --porcelain)" ]; then
  echo "Script must be applied on a clean git state"
  exit 1
fi

echo
echo "Checking formatting using the following black version:"
$black --version
echo

# perform python-format on all py-files
find ./src/torchlbm -name '*.py' -exec $black --line-length 160 {} \;
find ./src/mlbm -name '*.py' -exec $black --line-length 160 {} \;
find ./test -name '*.py' -exec $black --line-length 160 {} \;
# check if something was modified
black_notcorrectlist=`git status --porcelain | grep '^ M' | cut -c4-`
# if nothing changed ok
if [[ -z $black_notcorrectlist ]]; then
  # send a possitive message to gitlab
  echo "Excellent. Very good formatting!"
  exit 0;
else
  # Print all files that are not correct and how to manage those errors
  echo "The following files have pep8 format problems:"
  git diff --stat $black_notcorrectlist
  echo
  echo "Please run"
  echo
  echo "find ./src/torchlbm -name '*.py' -exec black --line-length 160 {} \;"
  echo "find ./src/mlbm -name '*.py' -exec black --line-length 160 {} \;"
  echo "find ./test -name '*.py' -exec black --line-length 160 {} \;"
  echo
  echo "Alternatively, use the Makefile and run following command from the TorchLBM basefolder:"
  echo
  echo "make format"
  echo
  # cleanup changes in git
  git reset HEAD --hard
fi

exit 1

