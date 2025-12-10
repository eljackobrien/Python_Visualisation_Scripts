# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:12:38 2022

@author: OBRIEJ25
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

save = 1

import IPython
shell = IPython.get_ipython()
shell.enable_matplotlib(gui='qt')

#%% Doppler effect visualisation

# Pre-define variables for circle and centres
num_frames = 2000
num_circs = 60
theta = ( np.linspace(-180, 180, 200) - 90) * np.pi/180
rx, ry = np.cos(theta), np.sin(theta)

# non-constant a -> no UVAST,    a = square_wave,    v = sawtooth,    t = time (linear)
periods = 2
velocity = 0.05 * signal.sawtooth( periods * np.linspace(0.5*np.pi, 2.5*np.pi, num_frames), width=0.5)
centres = np.zeros(num_frames)
centres[0] = -1
for t in range(num_frames):
    # new position is time step (=1) * velocity + previous position
    centres[t] = centres[t-1] + velocity[t]
centres = centres / abs(centres).max()
'''
plt.plot(np.gradient(velocity) / np.gradient(velocity).max(), label='Acceleration')
plt.plot(velocity / velocity.max(), label='Velocity')
plt.plot(centres / centres.max(), label='Position')
plt.legend(), plt.show() #'''

#centres = np.linspace(0, 2, num_frames)/1.5
radii = np.linspace(0, 18, num_frames)


#%% Make the animation
fig, ax = plt.subplots(1,1, figsize=(8,6))

plt.title("Doppler Effect ($\pm$Constant Acceleration)", fontsize=16)
plt.xlim(-5, 5), plt.ylim(-3.75, 3.75)
plt.xticks([]), plt.yticks([])

ax.set_facecolor('k')
plt.axhline(0, c='w', alpha=0.5, zorder=0, ls='--', lw=1.5)
plt.arrow(-0.7,-0.4, 1.5,0.0, color='w', width=0.05, head_width=0.17, head_length=0.14, length_includes_head=True)
plt.arrow(-0.0,-0.4, -0.8,0.0, color='w', width=0.05, head_width=0.17, head_length=0.14, length_includes_head=True)
#%%
# Define centre point and the desired circles. use plot, not scatter, as it returns the right iterable object
plot_centre, = ax.plot([], [], ls='', marker='o', markersize=8, c='m', zorder=10)
circs = [ax.plot([], [], lw=1.5, c='c', zorder=5)[0] for _ in range(num_circs)]

inds = num_frames*np.arange(0,num_circs)//num_circs
x0s = centres[ inds ]
alphas = np.hstack(( np.linspace(1, 0, num_frames//2), np.zeros(num_frames-num_frames//2) ))

# Function to update the data for each of the drawings, using set_data
def anim_func(frame):
    # Initialise circles
    if frame == 0:
        for circ in circs:
            circ.set_data([],[]), circ.set_alpha(1.0)

    plot_centre.set_data(centres[frame], 0)

    for circ, x0, ind in zip(circs, x0s, inds):
        if frame >= ind:
            circ.set_data([x0, *radii[frame-ind]*rx + x0],  [0, *radii[frame-ind]*ry])
            circ.set_alpha( np.roll( alphas, ind )[frame] )

fps = 120
anim = FuncAnimation(fig, anim_func, frames=len(centres), interval=1e3/fps)


if save: anim.save('doppler_effect_animation.mp4', writer='ffmpeg', fps=fps, dpi=300)
else: plt.show()

