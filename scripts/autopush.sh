#!/bin/bash
echo -e "commit message: \e[32m$0\e[m"
echo -e "target branch: \e[32m$1\e[m"

git add .
git commit -m "$0"
git push origin $1
