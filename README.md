# 2D Grid World - Collaborative Task Simulator

The 2D Grid World is a simulation environment designed to explore the dynamics of human-AI collaboration. It focuses on how AI can assist humans in completing tasks by considering the human's field of view (FOV), cognitive limitations, and task planning in a grid-based environment.

<p align="center">
  <img src="assets/game_play.png" width="80%" />
  <br> Example of a 2D Grid World layout with agents performing tasks while considering the human's field of view (tiles that are dimmed).
</p>


## Introduction

The 2D Grid World is a benchmark environment for cooperative human-AI interaction. Inspired by task-based collaborative games such as Overcooked, this environment simulates scenarios where humans and AI agents must work together to complete tasks like cooking, cleaning, or assembling items.

The key feature of this environment is the integration of a **limited Field of View (FOV)** for the human collaborator, which mimics the human's real-world perception constraints. The AI agent, therefore, must account for the human's restricted view when selecting actions, ensuring better task coordination.

### Key Features
- **Collaborative Task Planning**: Both the human and AI must complete multiple subtasks (e.g., cooking, cleaning) to achieve a shared goal.
- **Field of View Awareness**: The AI adapts its strategy by understanding and acting within the limited FOV of the human agent.
- **Procedural Task Layouts**: The environment supports customizable layouts to simulate various task settings, such as kitchens or workspaces.

## Steak Task Instructions

### Objective:
Prepare and deliver a steak dish with garnish (onion) using the least number of steps. The task requires completing the following subtasks: grilling the steak, chopping the onion, washing a plate, plating and serving the dish.

### Steps to Complete the Steak Task:
<p align="center">
  <img src="assets/game_interface.png" width="30%" />
  <img src="assets/game_instructions.png" width="50%" />
</p>


## Citation ##
Please cite this work using the following Bibtex:
```
Coming soon
```

## Contact ##
For any questions, please reach out to: [yachuanh@usc.edu](mailto:yachuanh@usc.edu)
