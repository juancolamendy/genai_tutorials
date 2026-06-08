from agno.workflow.workflow import Workflow
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.db.sqlite import SqliteDb
from agno.run import RunContext

db = SqliteDb(db_file="sessions/shared_state_workflow.db")

# Tools that operate on workflow session_state
def add_item(run_context: RunContext, item: str) -> str:
    if not run_context.session_state:
        run_context.session_state = {}
    if "shopping_list" not in run_context.session_state:
        run_context.session_state["shopping_list"] = []
    if item.lower() not in [x.lower() for x in run_context.session_state["shopping_list"]]:
        run_context.session_state["shopping_list"].append(item)
        return f"Added '{item}' to the shopping list."
    return f"'{item}' is already in the shopping list."

def list_items(run_context: RunContext) -> str:
    if not run_context.session_state:
        run_context.session_state = {}
    items = run_context.session_state.get("shopping_list", [])
    if not items:
        return "Shopping list is empty."
    return "Shopping list:\n" + "\n".join(f"- {i}" for i in items)

shopping_assistant = Agent(
    name="Shopping Assistant",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[add_item, list_items],
    debug_mode = True,
)

list_manager = Agent(
    name="List Manager",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[list_items],
    debug_mode = True,
)

shopping_workflow = Workflow(
    name="Shopping List Workflow",
    db=db,
    steps=[shopping_assistant, list_manager],
    session_state={"shopping_list": []},  # shared state
)

if __name__ == "__main__":
    shopping_workflow.print_response(
        input="Please add milk and bread, then show me the list."
    )
    print("Workflow session state:", shopping_workflow.get_session_state())

