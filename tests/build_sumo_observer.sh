#!/bin/sh
set -eu
mj=$(../sumo/.pixi/envs/default/bin/python -c 'import importlib.util; print(importlib.util.find_spec("mujoco").submodule_search_locations[0])')
cc -std=c11 -O2 -fPIC -shared -I"$mj/include" tests/sumo_observer.c -L"$mj" -Wl,-rpath,"$mj" -l:libmujoco.so.3.6.0 -ldl -o build/sumo_observer.so
