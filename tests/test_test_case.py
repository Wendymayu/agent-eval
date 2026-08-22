from agenteval.test_case import LLMTestCase, ConversationalTestCase, Turn, ToolCall


def test_llm_test_case_required_input_only():
    tc = LLMTestCase(input="What is 2+2?")
    assert tc.input == "What is 2+2?"
    assert tc.actual_output is None
    assert tc.expected_output is None


def test_llm_test_case_full_fields():
    tc = LLMTestCase(
        input="q",
        actual_output="4",
        expected_output="4",
        context=["2+2=4"],
        tools_called=[ToolCall(name="calc")],
        expected_tools=[ToolCall(name="calc")],
    )
    assert tc.tools_called[0].name == "calc"
    assert tc.multimodal is False


def test_conversational_test_case_turns():
    tc = ConversationalTestCase(
        turns=[Turn(role="user", content="hi"), Turn(role="assistant", content="hello")]
    )
    assert tc.turns[0].role == "user"
    assert len(tc.turns) == 2
