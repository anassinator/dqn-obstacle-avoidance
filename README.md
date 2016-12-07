# Deep Reinforcement Learning for Fixed-Wing Flight Control

This is a Deep Q-Netwrok reinforcement learning agent which navigates a fixed
wing aircraft in a simulator to a target waypoint while avoiding stationary
and moving obstacles.

This is our submission to our final project of McGill University's ECSE 526 -
Artificial Intelligence course.

<img width="551" alt="screen shot 2016-11-30 at 02 24 59" src="https://cloud.githubusercontent.com/assets/723610/20955252/b2f99780-bc0f-11e6-8bde-b441b763539f.png">


# Setup

To run this, one needs to set up MIT's director visualization tool as it will
be used to display the simulator. Instructions on how to build this can be
found [here](https://github.com/RobotLocomotion/director).

Following that, you must set up an alias for the `directorPython` executable
that was built. This can be simply done by running:

```bash
alias director=/path/to/director/build/install/bin/directorPython
```

You can add this to your shell profile to avoid running this every time.

Finally, you need to install Tensorflow for Python 2.7. This can be done by
following the steps
[here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html).

# Running

Once everything is setup, you can simply run:

```bash
director simulator.py
```

or

```bash
director simulator.py --help
```

for advanced options.

If everything works out, a window should appear with a plane model that flies
around slowly learning how to reach the green circle and avoiding the white
circle as in the screenshot above. The white circles denote obstacles, whereas
the green circle denotes a target waypoint. The rays protruding from the plane
represent the distances measured by the plane's sensor.

The learning process will take several hundreds of episodes, but rerunning the
simulation will proceed from where it left off by reusing the `model.ckpt` file.

# Acknowledgements

This project was inspired by the work found
[here](https://github.com/peteflorence/Machine-Learning-6.867-homework).
