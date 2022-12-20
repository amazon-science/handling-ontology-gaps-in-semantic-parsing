#!/usr/bin/env bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0



wget https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip -O data/TOPv2_Dataset.zip
mkdir data/TOPv2_Dataset
unzip data/TOPv2_Dataset.zip -d data/TOPv2_Dataset
rm data/TOPv2_Dataset.zip