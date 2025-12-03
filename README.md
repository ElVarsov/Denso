# EV Charging Scheduler

A small tool to schedule charging for multiple electric vehicles sharing a single charger. It finds fair, practical charging schedules based on each car's needs and availability, and provides a simple visual summary.

Key points

- Single charger, one car charges at a time.
- Goal: meet each car's commute needs while minimizing interruptions and balancing extra charging fairly.
- Comes with a basic Tkinter UI for adding cars and a plotting function for visual schedules.

Quick start

1. Install dependencies:
   - python 3.8+
   - pip install pulp matplotlib numpy
   - tkinter (usually preinstalled on desktop Python)
2. Run the UI:
   - From the project folder: python ui.py
   - Add cars, set arrival/departure and battery info, click "Optimize Schedule".
3. Run the example:
   - python ui.py will run a sample scenario and show a plot.

What to expect

- A visual timeline showing when each car charges.
- Per-car stats (minimum required charging, allocated time, estimated final state of charge).
- Simple controls in the UI for quick experimentation.

Where to look for details

- algorithm.py — actual scheduling logic, solver setup and visualization.
- ui.py — minimal UI for entering car data and running the optimizer.
- The code contains comments and short notes for tuning and debugging (solver logs, constraints, and common fixes).

Tips

- Reduce time number of cars for faster results.
- If the solver reports infeasible, check for overly strict availability or equal-allocation constraints.

Contributing / Next steps

- Ideas: multi-charger support, variable charging power, cost-aware scheduling.
- If you want a slide deck or a compact starter script to generate images, those can be added separately.

Contact

- Use the repository for issues or feature requests.
