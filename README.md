# An Enhanced Genetic Algorithm for Emergency Response

**Integrating Dynamic Fleet Modeling, Intersection Impedance, and Adaptive Counterflow Logic**

---

## 1. Overview
This repository contains the computational prototype and simulation framework for the study, *"An Enhanced Genetic Algorithm for Emergency Response: Integrating Dynamic Fleet Modeling, Intersection Impedance, and Adaptive Counterflow Logic."* Traditional emergency routing systems predominantly rely on static, distance-based shortest-path algorithms (e.g., Dijkstra's algorithm) that fail to account for stochastic urban gridlock, compounding fleet congestion, and environmental hazards. To address these systemic limitations, this repository introduces an Enhanced Genetic Algorithm (EGA) designed to optimize emergency vehicle dispatch. By integrating dynamic fleet cycles, intersection delay logic, and adaptive counterflow mechanisms, this simulation provides a resilient decision-support framework tailored for high-density urban environments, utilizing Cagayan de Oro City as the primary testing corridor.

## 2. Core Architectural Modules
The system architecture operates as a discrete-event simulation platform, functioning through four distinct, interconnected modules:

* **Data Input Layer (The Simulation Environment):** Ingests real-world geospatial data from OpenStreetMap (OSM) to construct a directed mathematical graph of the road network. It utilizes a programmatic scenario generator to inject fluctuating traffic volumes, peak-hour constraints, and environmental hazards (e.g., flooded nodes).
* **Dynamic Fleet Management Module:** Replaces static, proximity-based dispatching with a continuous-loop system. It programmatically tracks multi-trip vehicle availability, models turnaround/restocking times, and enables cross-station reassignment to prevent service denials during multi-casualty events.
* **The EGA Routing Engine (The Optimization Core):** The computational nucleus of the system. Instead of calculating a single definitive path, the EGA generates a diverse population of feasible route chromosomes, evolving them iteratively via tournament selection, uniform crossover, and adaptive mutation to converge upon the most temporally efficient path.
* **Simulation Interface and Telemetry:** A visualization and analytics layer that renders active vehicle trajectories on a Folium map interface, simultaneously logging key performance indicators (KPIs) such as Average Response Time (ART) and intersection impedance penalties.

## 3. Algorithmic Enhancements
The EGA distinguishes itself from baseline routing algorithms through the integration of a multi-objective fitness function and dynamic graph modifications:

* **Intersection Impedance Processing:** Intersections are not treated as zero-delay nodes. The algorithm employs a modified Bureau of Public Roads (BPR) congestion function to mathematically penalize routes passing through highly saturated junctions, forcing the EGA to evaluate the trade-off between physical distance and temporal delay.
* **Adaptive Counterflow Logic:** Under extreme gridlock, standard adherence to traffic directionality becomes a fatal constraint. The system incorporates a conditional logic gate that evaluates the risk-reward ratio of utilizing oncoming lanes. If activated, it reverses the directional vector of adjacent edges, applying a strict safety reduction factor (α = 0.60) to account for the necessary caution required by emergency operators.

## 4. Technical Stack
The computational prototype was engineered utilizing object-oriented programming paradigms to ensure modularity. The underlying source code leverages the following primary libraries:
* **`Python 3.x`**: Core programming language.
* **`OSMnx` & `NetworkX`**: For geospatial data extraction, topological mapping, and directed graph construction.
* **`DEAP` (Distributed Evolutionary Algorithms in Python):** For the formulation and execution of the genetic algorithm, including population initialization and evolutionary operators.
* **`Folium` & `Branca`**: For the programmatic generation of interactive, web-based simulation map dashboards.
* **`NumPy` & `Matplotlib`**: For statistical analysis and data visualization.

## 5. Usage and Execution
The simulation environment tests the algorithm against various empirical scenarios, including clear road conditions (Control), localized micro-gridlocks, and flood-prone nodes.

To execute the simulation prototype:
1. Ensure all dependencies are installed via `pip install -r requirements.txt`.
2. Run the main execution script to initialize the fleet objects and construct the computational graph from the localized `.osm` file.
3. The engine will sequentially compute the paths for the Baseline (Shortest-Path) algorithm and the EGA, comparing objective response times.
4. Output telemetry and interactive HTML maps will be generated in the root directory for post-simulation empirical analysis.

## 6. Academic Affiliation
This study was developed and submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science at the **University of Science and Technology of Southern Philippines (USTP)**, Cagayan de Oro City.

**Researchers:**
* James Darwen Barrion Bañas
* Carl Abecia Cañas
* Ralph Rhey Abao Lumigue
* Jaber Arsa Sangcopan

**Adviser:** Dr. Maricel A. Esclamado, PhD

## 7. Disclaimer
This simulation and its associated algorithms were conducted solely for academic and research purposes. The findings, analyses, and conclusions are based on a controlled computational environment. The results do not represent actual live operational performance of emergency agencies, nor should they be interpreted as official policies, procedures, or directives of any government institution, emergency response office, or public safety organization. The proposed Enhanced Genetic Algorithm is intended as a decision-support and research model only.
