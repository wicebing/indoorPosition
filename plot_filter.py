'''
====================================================
Applying the Kalman Filter with Missing Observations
====================================================

This example shows how one may apply :class:`KalmanFilter` when some
measurements are missing.

While the Kalman Filter and Kalman Smoother are typically presented assuming a
measurement exists for every time step, this is not always the case in reality.
:class:`KalmanFilter` is implemented to recognize masked portions of numpy
arrays as missing measurements.

The figure drawn illustrates the trajectory of each dimension of the true
state, the estimated state using all measurements, and the estimated state
using every fifth measurement.
'''
import numpy as np
import pylab as pl
from pykalman import KalmanFilter
from scipy.signal import savgol_filter


# specify parameters

n_timesteps = 70



np.random.seed(0)
x = np.linspace(0, 10, n_timesteps)
y = 2*np.sin(x) + np.random.normal(0, 0.5, n_timesteps)
yz = [2*np.sin(x),2*np.cos(x)] + np.random.normal(0, 1, n_timesteps)
yz = yz.T

for t in range(n_timesteps):
    if t % 6 == 0:
        temp = 5*np.random.randn()
        y[t] += temp
        y[t+1] += temp
        yz[t] += temp
        yz[t+1] += temp
        
# sample from model

random_state = np.random.RandomState(0)
transition_matrices = [[1,1],[1,1]]
transition_offset = [0,0]
transition_covariance = [[0.1, 0.1], [0.1, 0.2]]
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0
observation_offset = [0,0]
observation_covariance = [[1]]
initial_state_mean = yz[0]

kf2 = KalmanFilter(
    transition_matrices=transition_matrices,
    observation_matrices=observation_matrix,
    transition_offsets=transition_offset,
    observation_offsets=observation_offset,
    initial_state_mean=initial_state_mean,
    random_state=0
)

kf = KalmanFilter(n_dim_obs=1,observation_matrices=[1])

# estimate state with filtering and smoothing
smoothed_states_missing = kf.smooth(y)[0]

# draw estimates
pl.figure()
lines_true = pl.plot(y, color='b')
lines_smooth_missing = pl.plot(smoothed_states_missing, color='g')
pl.legend(
    (lines_true[0], lines_smooth_missing[0]),
    ('true_1', 'missing_1'),
    loc='lower right'
)
pl.show()


smoothed_states_missing_yz = kf2.smooth(yz)[0]
# draw estimates
pl.figure()
lines_true = pl.plot(yz[:,0], color='b')
lines_smooth_missing = pl.plot(smoothed_states_missing_yz[:,0], color='g')
pl.legend(
    (lines_true[0], lines_smooth_missing[0]),
    ('true', 'missing'),
    loc='lower right'
)
pl.show()

# # 执行Savitzky-Golay滤波
# window_length = 20  # 窗口长度（奇数）
# polyorder = 2  # 多项式阶数
# y_smoothed = savgol_filter(y, window_length, polyorder)
# # draw estimates
# pl.figure()
# lines_true = pl.plot(y, color='b')
# lines_smooth_missing = pl.plot(y_smoothed, color='r')
# pl.legend(
#     (lines_true[0], lines_smooth_missing[0]),
#     ('true', 'missing'),
#     loc='lower right'
# )
# pl.show()
