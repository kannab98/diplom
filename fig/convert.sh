#!/bin/bash
for pdfile in *.pdf ; do
  convert    -density 300 "${pdfile}" "${pdfile%.*}".png
done
