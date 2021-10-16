from src.preprocessors.abstract import Preprocessor


def test_cleanup():
    pass


def test_remove_url():
    text = "'By accident' they knew what was gon happen https://t.co/Ysxun5vCeh"
    result = Preprocessor.remove_url(text)
    expected = "'By accident' they knew what was gon happen "

    assert result == expected
