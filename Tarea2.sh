#!/bin/bash

clear
echo "Executing CV Tarea2 bash file..."

git clone https://github.com/fjean/pymeanshift

cd pymeanshift

./setup.py install

cd ..

./Tarea2.py
