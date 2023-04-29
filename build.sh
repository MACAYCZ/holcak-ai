#!/bin/sh
set -ex
gcc -O3 -Wall -Wextra -pedantic -pthread -o build $(find -L tests/ -type f -name '*.c') -lm
