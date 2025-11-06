# Greenhouse Gas Visualization (2001–2025)
A parallel visualization of greenhouse gas (GHG) concentrations across time, built with MPI and Pygame. The project correlates with NOAA datasets (CO2, CH4, N2O) and visualizes their changing intensities as dynamic particles orbiting around Earth.

This project was designed to run on a 4-node MPI cluster driving a 2x2 display grid (four monitors).  
It can also be tested locally on a single machine using the provided local version.

---

##  Features

- Visualizes CO2, CH4, and N2O trends from 2001–2025  
- Parallelized using MPI (mpi4py) to render across multiple display nodes  
- Animated particle systems representing gas concentrations  
- Dynamic year progression  
- Scalable for local or multi-node visualization setups

---

##  Project Structure

ghg_visualization.py    
requirements.txt    
README.md   
LICENSE 

---

# Running the Project

Run across all nodes in your cluster:

```bash
mpirun --pernode --hostfile hosts_exec python3 ghg_visualization.py
```

--pernode ensures one process per node

--hostfile hosts_exec specifies your cluster node list

Each node displays one quadrant of the full visualization

Example setup:
Four monitors arranged in a 2x2 grid → each running one MPI process window.

## Data Sources
Data obtained from NOAA Global Monitoring Laboratory (GML):

* CO2 Trends
* CH4 Trends
* N2O Trends

⚙️ Requirements
Install dependencies via:

```bash
pip install -r requirements.txt
```

## Skills Demonstrated
* Parallel programming with MPI (mpi4py)
* Real-time rendering with Pygame
* Data visualization from CSV datasets
* Synchronization of distributed displays
* Data handling with pandas and NumPy

## License
MIT License (see LICENSE)

## Development Note
All code was developed, reviewed, and tested by me.
Used ChatGPT-5 as a coding assistant for learning purposes
