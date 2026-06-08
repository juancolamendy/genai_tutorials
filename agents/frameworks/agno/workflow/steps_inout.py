from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow
from agno.workflow.condition import Condition
from agno.agent import Agent
from agno.utils.pprint import pprint_run_response

# Define the research agent
research_agent = Agent(
    name="Research Agent",
    model="openai:gpt-4o-mini",
    instructions="Research and provide detailed information about the given topic. Be thorough and informative.",
    debug_mode=True,
)

# Create the research step
research_agent_step = Step(
    name="research",
    description="Research the given topic",
    agent=research_agent,
)

# Define a function-based summarizer step
def summarizer_step(step_input: StepInput, session_state) -> StepOutput:
    last_content = step_input.previous_step_content
    message = step_input.input
    # You could also access session_state here if needed
    summary = f"Message: {message} - Summary of previous step:\n{last_content[:500]}"
    return StepOutput(content=summary)

# Create the summary step using the function
summary_step = Step(
    name="summarize",
    description="Summarize previous agent output",
    executor=summarizer_step,
)

# Define a custom function step that updates session_state
def custom_function_step(step_input: StepInput, session_state):
    session_state["test"] = "test_1"  # update shared state
    return StepOutput(content=f"Updated session_state: {session_state}")

custom_step = Step(
    name="custom_step",
    description="Update session state with test value",
    executor=custom_function_step,
)

# Define an evaluator function for the condition
def evaluator_function(step_input: StepInput, session_state):
    return session_state.get("test") == "test_1"

# Define a display step that shows the session_state
def display_session_state(step_input: StepInput, session_state):
    return StepOutput(content=f"Session state contents: {session_state}")

display_step = Step(
    name="display_state",
    description="Display the current session state",
    executor=display_session_state,
)

# Create a condition step (evaluates if test == "test_1")
condition_step = Condition(
    name="condition_step",
    evaluator=evaluator_function,
    steps=[display_step],  # This step will run if condition is True
)

# Create the workflow
workflow = Workflow(
    name="Research + Summary",
    steps=[research_agent_step, summary_step, custom_step, condition_step],
    debug_mode=True,
)

# Run the workflow
if __name__ == "__main__":
    # Initialize session_state
    initial_session_state = {
        "workflow_name": "Research + Summary Demo",
        "started_at": "2026-02-06"
    }

    result = workflow.run(
        "Explain AI trends in 2026",
        session_state=initial_session_state
    )
    pprint_run_response(result, markdown=True)

    # Print final session state
    print("\n" + "="*50)
    print("Final Session State:")
    print("="*50)
    print(result.session_state if hasattr(result, 'session_state') else "No session state available")
