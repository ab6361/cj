import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some example data
x = np.linspace(0, 2 * np.pi, 100)

# Create a figure and axis
fig, ax = plt.subplots()

# The comma after line indicates that it is a single-element tuple.
line, = ax.plot(x, np.sin(x))

# Function to update the plot. frame is a variable that represents the current frame of an animation.
def update(frame):
    # Generate new data for each frame
    y = np.sin(x + 2 * np.pi * frame / 10)
    # Update the y coordinate of the line
    line.set_ydata(y)
    # Return the object being animated
    return line,

# Create the animation.
# The first parameter represents the Figure object that the animation will be associated with. The Figure object is responsible for managing the visual elements of the plot or graph.
# The second parameter the function that will be called at each frame of the animation. The update function should have a specific signature, taking the current frame as the first argument 
# 3rd argument is the number of frames in the animation
# blit improves the performance of the animation
ani = FuncAnimation(fig, update, frames = 10, blit = True)

# Display the animation
plt.show()
