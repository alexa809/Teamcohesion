#!/usr/bin/env python3
"""
CrewAI Puzzle Cohesion Simulation
Team: United Front
"""

def check_dependencies():
    try:
        from crewai import Agent, Task, Crew
        from langchain_ollama import OllamaLLM
        return True
    except ImportError:
        print("Error: Required Python packages are not installed.")
        print("Please run: pip install -r requirements.txt")
        return False

def check_ollama():
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("\nError: Ollama is not installed.")
        print("Please install Ollama from: https://ollama.com")
        print("After installing, run: ollama pull llama2")
        return False

if not check_dependencies() or not check_ollama():
    raise SystemExit(1)

# Import after checks pass
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

# Initialize LLM
llm = OllamaLLM(
    model="llama2",
    temperature=0.9,
    base_url="http://localhost:11434",
    verbose=True
)

# Agent Definitions
InteriorPiecePlacer = Agent(
    role="Interior Piece Placer",
    goal=(
        "Strategically place interior pieces to complete the puzzle with precision and efficiency, "
        "while keeping the team motivated."
    ),
    backstory=(
        "You're a seasoned puzzle master who instantly sees where each piece belongs. "
        "You boost morale through expert guidance."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

ExteriorPiecePlacer = Agent(
    role="Exterior Piece Placer",
    goal=(
        "Securely place exterior pieces to form a precise, stable border that guides subsequent interior placements."
    ),
    backstory=(
        "You instantly spot edge shapes and patterns, setting a foundation for the puzzle frame."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

PieceGrabber = Agent(
    role="Piece Grabber",
    goal=(
        "Identify and retrieve the most relevant pieces from the pool, minimizing search time "
        "and preventing bottlenecks."
    ),
    backstory=(
        "You have an uncanny ability to spot the right piece in a sea of shapes and colors, "
        "keeping workflow smooth."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

Planner = Agent(
    role="Planner",
    goal=(
        "Develop and communicate an efficient step-by-step plan that balances progress across the puzzle "
        "and synchronizes the team."
    ),
    backstory=(
        "You're the master architect, always thinking several moves ahead and guiding strategic phases of the puzzle."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Task Definitions
interior_piece_placer_task = Task(
    description=(
        "Strategically place interior puzzle pieces to fill in the core of the puzzle, "
        "guiding the team through tricky fits."
    ),
    expected_output=(
        "All interior sections are completed accurately with minimal misalignments "
        "and high team engagement."
    ),
    agent=InteriorPiecePlacer
)

exterior_piece_placer_task = Task(
    description=(
        "Place all border and edge pieces to form the puzzle frame, "
        "creating a clear boundary for the interior."
    ),
    expected_output=(
        "A fully assembled puzzle border matching the image edges, providing a stable outline."
    ),
    agent=ExteriorPiecePlacer
)

piece_grabber_task = Task(
    description=(
        "Quickly identify and hand off the correct puzzle pieces requested by the placers, "
        "minimizing search time."
    ),
    expected_output=(
        "Placers receive the right piece on demand, leading to smooth placement flow."
    ),
    agent=PieceGrabber
)

planner_task = Task(
    description=(
        "Develop and communicate a strategic plan for solving the puzzle, "
        "breaking it into logical segments."
    ),
    expected_output=(
        "Team follows a clear plan, progress is well-coordinated, and sections are completed efficiently."
    ),
    agent=Planner
)

# Optional: Bonding Task for variable group
dinner_bonding_task = Task(
    description=(
        "Team bonds over a dinner before puzzle-solving, enhancing cohesion and communication."
    ),
    expected_output=(
        "Team reports increased cohesion and readiness to tackle the puzzle."
    ),
    agent=Planner  # any agent can facilitate
)

# Run Simulations
def run_group(name: str, agents: list, tasks: list) -> None:
    """
    Run a simulation with the given group of agents and tasks.
    
    Args:
        name (str): Name of the simulation group (Control or Variable)
        agents (list): List of Agent objects to participate in the simulation
        tasks (list): List of Task objects to be completed
    """
    print(f"\n{'='*50}")
    print(f"{name} Simulation Starting")
    print(f"{'='*50}")
    print(f"Total tasks: {len(tasks)}")
    
    # Initialize progress tracking
    total_tasks = len(tasks)
    completed_tasks = 0
    last_task = ""
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )
    
    # Start simulation
    result = crew.kickoff()
    
    for msg in result.interactions:
        speaker = msg.get('speaker', 'Unknown')
        text = msg.get('text', '')
        
        # Check for task completion indicators
        if any(keyword in text.lower() for keyword in ['completed', 'finished', 'done']):
            if text != last_task:  # Avoid counting the same task twice
                completed_tasks += 1
                last_task = text
                
                # Calculate progress
                progress = completed_tasks / total_tasks
                bar_length = 30
                filled_length = int(bar_length * progress)
                
                # Create progress bar
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                
                # Clear previous line and print progress
                print(f"\r\033[K[{bar}] {completed_tasks}/{total_tasks} tasks completed ({progress*100:.0f}%)", end='')
        
        # Print agent messages
        print(f"\n[{speaker}] {text}")
    
    # Print final results
    cohesion = result.controlCohesion if name == 'Control' else result.variableCohesion
    print(f"\n\nFinal Cohesion Score: {cohesion}%")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    agents = [
        InteriorPiecePlacer,
        ExteriorPiecePlacer,
        PieceGrabber,
        Planner
    ]

    # Control group: only puzzle tasks
    control_tasks = [
        interior_piece_placer_task,
        exterior_piece_placer_task,
        piece_grabber_task,
        planner_task
    ]
    run_group('Control', agents, control_tasks)

    # Variable group: bonding + puzzle tasks
    variable_tasks = [dinner_bonding_task] + control_tasks
    run_group('Variable', agents, variable_tasks)
