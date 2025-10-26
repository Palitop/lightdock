"""Tests for IO class"""

from lightdock.ioutil.IO import IO


class TestIO:
    def test_cstrip(self):
        assert IO.cstrip("  TEST  ") == "TEST"
        assert IO.cstrip("TEST") == "TEST"
        assert IO.cstrip("   ") == ""
        assert IO.cstrip("") == ""
