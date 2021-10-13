from src.prepare import SKCountVectorizer


def test_cleanup():
    text = "'By accident' they knew what was gon happen https://t.co/Ysxun5vCeh"
    obj = SKCountVectorizer()
    result = obj.cleanup(text)
    expected = "'By accident' they knew what was gon happen "

    assert result == expected
