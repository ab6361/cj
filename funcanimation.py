import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function to animate
def func(x, k):
    return x**2 / k

# Generate x values
x = np.linspace(-10, 10, 100)

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, func(x, 1))  # Initial plot with k=1

# Set axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Animation of f(x) = x^2 / k')

# Set the axis limits
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 100)

# Update function for the animation
def update(k):
    y = func(x, k)
    line.set_ydata(y)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 11), blit=True)

# Display the animation
plt.show()