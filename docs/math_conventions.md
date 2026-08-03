# Mathematical conventions

The entire repository uses one direction:

- `t=0`: Gaussian prior/noise;
- `t=1`: MRI data;
- `x_t = (1-t) z + t x_data`;
- target velocity `v = x_data - z`;
- sampling integrates `dx/dt = v_theta(x,t)` from 0 to 1.

The FreeFlow paper writes noise at time 1 and data at time 0. Track A translates its duration
variable to this repository convention. A prior-anchored prediction at duration `delta` therefore
queries the teacher at `t=delta`, not `1-delta`.

For correction, `x_r=(1-r)y_hat+r n` uses `r=0` at generated data and `r=1` at fresh noise. The
equivalent teacher time in this repository is `t=1-r`.

