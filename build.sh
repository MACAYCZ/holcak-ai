#!/bin/sh
set -ex
gcc -O3 -Wall -Wextra -pedantic -o build $(find -L tests/ -type f -name '*.c') -lpthread -lm
