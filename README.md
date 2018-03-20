# Summary

* This is a simple matrix transpose
* Use the ARM neon SIMD instrinsics & prefetch
* Understand the impact of software prefetch


# Build
* `$ make`

# Result
```
./main_arm_pre
neon_pre:     969119 us

./main_arm
neon:        1672894 us
```

# Test Performance of Prefetching with Multi Thread Version
```
// multi thread
git checkout multi-thread
make prefetch

// multi thread + loop unrolling
git checkout multi-thread_loop_unrolling
make prefetch
```

# License
`NCTU_embedded_prefetch` is freely redistributable under the two-clause BSD License.
Use of this source code is governed by a BSD-style license that can be found
in the `LICENSE` file.
