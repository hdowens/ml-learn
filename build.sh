#!/bin/sh

set -xe
clang -o -Wall -Wextra -o gates gates.c -lm 
clang -o -Wall -Wextra -o twice twice.c -lm