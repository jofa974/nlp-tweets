from src.prepare import cleanup


def test_cleanup():
    text = "'By accident' they knew what was gon happen https://t.co/Ysxun5vCeh"
    result = cleanup(text)
    expected = "'By accident' they knew what was gon happen "

    assert result == expected
