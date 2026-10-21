| scenario    | controller   | tuning   |   min_gap_m |   min_time_gap_s |   max_jerk_ms3 (noisy) |   rms_accel_ms2 |   rms_gap_error_m | collision   |
|:------------|:-------------|:---------|------------:|-----------------:|-----------------------:|----------------:|------------------:|:------------|
| gentle      | PID          | good     |       24.67 |             1.84 |                   7.00 |            0.57 |              0.43 | no          |
| gentle      | PID          | pd       |       26.84 |             2.04 |                   1.14 |            0.53 |              2.51 | no          |
| gentle      | MPC          | good     |       25.21 |             1.91 |                  16.40 |            1.52 |              1.07 | no          |
| emergency   | PID          | good     |        7.60 |             1.84 |                  53.55 |            2.34 |              1.32 | no          |
| emergency   | PID          | pd       |        8.09 |             2.04 |                  56.78 |            2.30 |              2.32 | no          |
| emergency   | PID          | bad      |        8.60 |             2.10 |                  29.40 |            1.88 |              9.37 | no          |
| emergency   | MPC          | good     |        8.17 |             1.91 |                  63.06 |            2.26 |              1.09 | no          |
| emergency   | MPC          | bad      |        9.97 |             2.44 |                   6.34 |            1.63 |              8.81 | no          |
| stop_and_go | PID          | good     |        8.63 |             2.07 |                  31.28 |            1.29 |              0.88 | no          |
| stop_and_go | PID          | pd       |        9.18 |             2.37 |                  30.29 |            1.26 |              1.81 | no          |
| stop_and_go | PID          | bad      |        8.57 |             2.99 |                  29.64 |            1.23 |              8.19 | no          |
| stop_and_go | MPC          | good     |        8.75 |             2.17 |                  42.21 |            1.58 |              0.77 | no          |
| stop_and_go | MPC          | bad      |       11.99 |             2.84 |                   4.21 |            1.02 |              6.24 | no          |