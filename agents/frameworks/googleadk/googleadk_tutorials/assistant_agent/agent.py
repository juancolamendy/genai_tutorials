from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from datetime import datetime
from typing import Dict, Any, List

def save_user_preference(
    preference_type: str, 
    value: str, 
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Save a user preference with timestamp.
    
    Args:
        preference_type: Type of preference (e.g., 'cuisine', 'music_genre')
        value: The preference value
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Operation status and details
    """
    # Store preference with user scope
    preference_key = f"user:preference_{preference_type}"
    timestamp_key = f"user:preference_{preference_type}_updated"
    
    # Save the preference and when it was set
    tool_context.state[preference_key] = value
    tool_context.state[timestamp_key] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "message": f"Saved {preference_type} preference: {value}",
        "updated_at": tool_context.state[timestamp_key]
    }

def get_user_profile(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieve comprehensive user profile information.
    
    Args:
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: User profile data including preferences and history
    """
    # Get user name
    user_name = tool_context.state.get("user:name", "Guest")
    
    # Collect all user preferences
    preferences = {}
    
    # Check for common preference types
    common_preferences = [
        "cuisine", "music_genre", "favorite_color", "language", "outdoor_activity",
        "timezone", "notification_preference", "theme", "accessibility", "reading_genre"
    ]
    
    for pref_type in common_preferences:
        pref_key = f"user:preference_{pref_type}"
        if pref_key in tool_context.state:
            preferences[pref_type] = tool_context.state[pref_key]
            # Get timestamp if available
            timestamp_key = f"{pref_key}_updated"
            if timestamp_key in tool_context.state:
                preferences[f"{pref_type}_updated"] = tool_context.state[timestamp_key]
    
    # Get conversation history
    last_interaction = tool_context.state.get("last_interaction", "none")
    total_interactions = tool_context.state.get("user:total_interactions", 0)
    
    return {
        "user_name": user_name,
        "preferences": preferences,
        "last_interaction": last_interaction,
        "total_interactions": total_interactions,
        "profile_retrieved_at": datetime.now().isoformat()
    }

def track_conversation_flow(
    flow_type: str,
    step: str,
    data: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Track multi-step conversation flows for better context.
    
    Args:
        flow_type: Type of flow (e.g., 'booking', 'planning', 'troubleshooting')
        step: Current step in the flow
        data: Relevant data for this step
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Flow tracking status and current state
    """
    # Store flow information
    flow_key = f"user:flow_{flow_type}"
    step_key = f"user:flow_{flow_type}_step"
    data_key = f"user:flow_{flow_type}_data"
    timestamp_key = f"user:flow_{flow_type}_updated"
    
    tool_context.state[flow_key] = flow_type
    tool_context.state[step_key] = step
    tool_context.state[data_key] = data
    tool_context.state[timestamp_key] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "message": f"Tracked {flow_type} flow: {step}",
        "current_step": step,
        "flow_data": data,
        "updated_at": tool_context.state[timestamp_key]
    }

def update_user_interaction(
    interaction_type: str,
    details: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Update user interaction history and preferences.
    
    Args:
        interaction_type: Type of interaction (e.g., 'question', 'request', 'feedback')
        details: Details about the interaction
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Interaction update status
    """
    # Update interaction count
    current_count = tool_context.state.get("user:total_interactions", 0)
    tool_context.state["user:total_interactions"] = current_count + 1
    
    # Store interaction details
    interaction_key = f"user:interaction_{current_count + 1}"
    tool_context.state[interaction_key] = {
        "type": interaction_type,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    # Update last interaction
    tool_context.state["last_interaction"] = f"{interaction_type}: {details}"
    
    return {
        "status": "success",
        "message": f"Updated interaction: {interaction_type}",
        "interaction_count": current_count + 1,
        "timestamp": datetime.now().isoformat()
    }

# Tools list
state_tools = [
    save_user_preference, 
    get_user_profile, 
    track_conversation_flow,
    update_user_interaction
]

# Personal assistant with state awareness
root_agent = Agent(
    name="personal_assistant",
    model="gemini-3-flash-preview",
    instruction="""
    You are a highly personalized assistant that remembers user preferences and context.
    
    STARTUP BEHAVIOR:
    - Always check user state at the beginning of each interaction
    - If user:name exists, greet them by name
    - If this is a returning user, reference relevant previous preferences
    - Check for any ongoing conversation flows and offer to continue them
    
    STATE USAGE GUIDELINES:
    - Use save_user_preference tool when users express preferences
    - Use get_user_profile tool to understand user background before making recommendations
    - Use track_conversation_flow for multi-step processes (booking, planning, troubleshooting)
    - Use update_user_interaction to track user engagement and build context
    
    PERSONALIZATION:
    - Tailor responses based on user:preferences
    - Reference previous interactions when relevant
    - Maintain consistency with established user relationships
    - Learn from user feedback and adjust recommendations accordingly
    
    MEMORY MANAGEMENT:
    - Store important decisions and outcomes
    - Remember user goals and aspirations
    - Track what works well for each user
    - Maintain conversation context across sessions
    
    CONVERSATION FLOW:
    - For new users, focus on learning their preferences
    - For returning users, reference their history and preferences
    - Proactively suggest improvements based on past interactions
    - Handle multi-step processes with clear progress tracking
    """,
    tools=state_tools,
    output_key="last_assistant_response"
)
