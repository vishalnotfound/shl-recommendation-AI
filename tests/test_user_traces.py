import pytest
from app.models import ChatRequest, Message
from app.agent import handle_chat
import asyncio

@pytest.mark.asyncio
async def test_trace_1_graduate_financial_analysts():
    """
    Test Trace 1: Graduate Financial Analysts
    User provides role, seniority, and specific test types in the first turn.
    Expected: Agent should immediately recommend.
    """
    request = ChatRequest(messages=[
        Message(role="user", content="Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.")
    ])
    
    response = await handle_chat(request)
    
    # Assert it recommends instead of clarifying
    assert len(response.recommendations) > 0, "Agent should have provided recommendations."
    
    # Verify the recommended items match what was requested
    rec_names = [r.name.lower() for r in response.recommendations]
    assert any("numerical reasoning" in name or "finance" in name for name in rec_names), "Should recommend numerical/finance tests."


@pytest.mark.asyncio
async def test_trace_senior_leadership_development_loop_break():
    """
    Test the specific sequence where the user kept answering vaguely.
    Our recent 'loop-breaker' rule should eventually force a recommendation.
    """
    request = ChatRequest(messages=[
        Message(role="user", content="We need a solution for senior leadership."),
        Message(role="assistant", content="What specific senior leadership position are you looking to assess?"),
        Message(role="user", content="The pool consists of CXOs, director-level postions; people with more than 15 years of experience."),
        Message(role="assistant", content="Is this assessment for selection or development purposes?"),
        Message(role="user", content="development")
    ])
    
    response = await handle_chat(request)
    
    # Assert it breaks the clarification loop and provides recommendations
    assert len(response.recommendations) > 0, "Agent should break the clarify loop and provide recommendations for long conversations."


@pytest.mark.asyncio
async def test_trace_team_leadership_no_personality():
    """
    Test the trace where the user said "no" to personality tests.
    Ensures that "no" doesn't cause a Pydantic crash (None validation error).
    """
    request = ChatRequest(messages=[
        Message(role="user", content="We need a solution for senior leadership."),
        Message(role="assistant", content="What specific senior leadership position are you looking to assess?"),
        Message(role="user", content="The pool consists of CXOs, director-level postions; people with more than 15 years of experience."),
        Message(role="assistant", content="Is this assessment for selection or development purposes?"),
        Message(role="user", content="Selection"),
        Message(role="assistant", content="Could you tell me more about the role and what specific leadership competencies you need to evaluate?"),
        Message(role="user", content="team leadereship"),
        Message(role="assistant", content="Do you want to include personality assessments in the evaluation of team leadership competencies?"),
        Message(role="user", content="no")
    ])
    
    response = await handle_chat(request)
    
    # Assert it doesn't crash into FALLBACK_CLARIFY, and instead recommends
    assert len(response.recommendations) > 0, "Agent should handle 'no' correctly without crashing and provide recommendations."
