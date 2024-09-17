# 2D Grid World - Collaborative Task Simulator

The 2D Grid World is a simulation environment designed to explore the dynamics of human-AI collaboration. It focuses on how AI can assist humans in completing tasks by considering the human's field of view (FOV), cognitive limitations, and task planning in a grid-based environment.

![2D Grid World Layout](assests/env.png)
Example of a 2D Grid World layout with agents performing tasks while considering the human's field of view (tiles that are dimmed).

## Introduction

The 2D Grid World is a benchmark environment for cooperative human-AI interaction. Inspired by task-based collaborative games such as Overcooked, this environment simulates scenarios where humans and AI agents must work together to complete tasks like cooking, cleaning, or assembling items.

The key feature of this environment is the integration of a **limited Field of View (FOV)** for the human collaborator, which mimics the human's real-world perception constraints. The AI agent, therefore, must account for the human's restricted view when selecting actions, ensuring better task coordination.

### Key Features
- **Collaborative Task Planning**: Both the human and AI must complete multiple subtasks (e.g., cooking, cleaning) to achieve a shared goal.
- **Field of View Awareness**: The AI adapts its strategy by understanding and acting within the limited FOV of the human agent.
- **Procedural Task Layouts**: The environment supports customizable layouts to simulate various task settings, such as kitchens or workspaces.

## Steak Task Instructions

### Objective:
Prepare and deliver a steak dish with garnish (onion) using the least number of steps. The task requires completing the following subtasks: grilling the steak, chopping the onion, washing a plate, and serving the final dish.

### Steps to Complete the Steak Task:

1. **Pick up the raw steak:**
   - Navigate to the **meat dispenser** on the grid.
   - Use the interaction action to **pick up the raw steak**.

2. **Grill the steak:**
   - Move to the **grill station** with the raw steak.
   - Place the steak on the grill using the interaction action.
   - **Wait for 10 timesteps** for the steak to cook.

3. **Pick up the onion:**
   - While the steak is grilling, navigate to the **onion dispenser**.
   - Use the interaction action to **pick up an onion**.

4. **Chop the onion:**
   - Move to the **chopping board**.
   - Use the interaction action to place the onion on the chopping board and chop it. This will take a few timesteps.

5. **Wash a plate:**
   - Navigate to the **plate dispenser** to pick up a dirty plate.
   - Move to the **sink** to wash the plate. Use the interaction action to clean the plate.

6. **Pick up the cooked steak:**
   - Return to the grill station where the steak has finished cooking.
   - **Pick up the steak** using the interaction action.

7. **Assemble the dish:**
   - Move to the **chopping board** to pick up the chopped onion garnish.
   - **Combine** the steak and onion on the plate.

8. **Serve the steak dish:**
   - Finally, take the **completed steak dish** to the **serving counter** and deliver it using the interaction action.

<p align="center">
  <img src="assets/game_instructions.png" width="50%" />
</p>


## Citation ##
Please cite this work using the following Bibtex:
```
Coming soon
```

## Contact ##
For any questions, please reach out to: [yachuanh@usc.edu](mailto:yachuanh@usc.edu)
