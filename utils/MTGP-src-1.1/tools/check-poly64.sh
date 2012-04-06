#!/bin/bash
for i in 11213 23209 44497; do
  for ((j=0;j<=127;j++)) do
    ./check-poly 32 $i $j;
  done
done
