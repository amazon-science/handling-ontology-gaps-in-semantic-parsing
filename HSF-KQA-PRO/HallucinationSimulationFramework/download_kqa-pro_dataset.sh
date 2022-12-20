#!/usr/bin/env bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0


final_url=$(wget -O /dev/null "https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1" 2>&1 | grep -w 'Location')

url=$(python3 -c "print('$final_url'.split(' ')[1])")

wget $url -O data/KQAPro.IID.zip


unzip data/KQAPro.IID.zip -d data/
rm -rf data/KQAPro.IID.zip