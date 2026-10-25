# Rope on a table — Elastica++ case study

An Elastica++ (`GazzolaLab/elasticapp`) case study: a Cosserat rod on a
frictional plane, one end clamped, the free end dragged along a prescribed
path. Renders through the same POVray pipeline as the PyElastica version.

## Where the files go

Elastica++ is a C++ library built from source — case studies live inside the
repo tree, not in your own project directory:

```
elasticapp/examples/rope_on_table/
    CMakeLists.txt
    RopeOnTableCaseStudy.hpp
    RopeOnTableCaseStudy.cpp
    visualize.py
    cli -> ../cli          # symlink, same as every other example
```

Then register it in `examples/CMakeLists.txt`, alongside the other case
studies:

```cmake
add_subdirectory(rope_on_table)
```

```bash
cd elasticapp/examples/rope_on_table && ln -s ../cli cli
```

## Build

Needs a C++20 compiler, CMake, Ninja, and vcpkg:

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake -B build -G Ninja \
  -D CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -D BUILD_TESTING=OFF \
  -D ELASTICA_BUILD_BENCHMARKS=OFF \
  -D ELASTICA_BUILD_EXAMPLES=ON \
  -D ELASTICA_BUILD_PYTHON_BINDINGS=ON
cmake --build build --parallel

cd build/bin/python && python3 -m pip install --editable .
```

That produces two targets: a standalone `example_rope_on_table` binary and the
`PyRopeOnTableCaseStudy` Python module.

## Run

```bash
python visualize.py --drive lift --final-time 3.0   # -> rope_on_table.dat
python rope_render.py --data rope_on_table.dat --still 1.0 --width 2400 --height 1600
```

`rope_render.py`, `_povmacros.py`, `_povextras.py` and `table.inc` come from
the PyElastica POVray package — the `.dat` schema is identical, so nothing
there changes.

## Design notes

* **Frame**: the case study is z-up (table is `z = 0`, gravity along `-z`),
  matching `rod_plane_collision`. `visualize.py` swaps axes to y-up on export
  because POVray is y-up. Pass `--up z` to keep the simulation frame.
* **Friction** is a property of the *material*, not of a separate forcing
  object as in PyElastica. `create_material` takes static and kinetic
  coefficients, and the collision system reads them off the contacting pair.
* **The table** is an `ImposedPlanePrimitive`. Even though it never moves it
  still needs a kinematics buffer, so all three axes get zero position and
  velocity profiles.
* **Constraints** replace PyElastica's boundary-condition classes.
  `make_value_constraint` sets positions/directors and `make_rate_constraint`
  sets velocities; both receive the current time, so the tip drive is just a
  lambda of `t`. The tip constraint is applied `.at(elastica::end - 1UL)` —
  the last *element* — and indexes `index + 1UL` for the last *node*, the same
  convention `helical_buckling` uses.
* **The tip velocity** is a central difference of the position path rather
  than a hand-derived analytic form, so the value and rate constraints can
  never drift out of sync when the profile is edited.

## Caveat

Unlike the PyElastica version, this code has **not been compiled or run** — a
full vcpkg + C++20 toolchain build wasn't available on my end. The API usage is
taken directly from the repo's own examples (`rod_plane_collision/plane` for
the plane, collision and material setup; `flexible_swinging_pendulum` for
clamping and the Python driver; `helical_buckling` for time-varying
constraints), but expect to fix compile errors on first build — most likely
around the `Rot3` construction for the clamped director or the exact spelling
of the `elasticapp` Python module path.

If you only need figures for the paper, the PyElastica + POVray version is the
lower-risk path: same renderer, same output, no build system. Elastica++ is
worth it when you need the speed, which matters for large Koopman data
collection sweeps rather than for a single figure.
