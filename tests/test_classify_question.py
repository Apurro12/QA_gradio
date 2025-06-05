from use_cases.classify_question import classify_question

def test_classify_question():
    result = classify_question("Sample question?")
    assert result == "No documents found."