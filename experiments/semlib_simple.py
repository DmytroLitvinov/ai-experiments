"""
Based on article https://anishathalye.com/semlib/
"""

from semlib.map import map
from semlib.sort import sort as sorted
from semlib.extrema import max


rs = ["Ilya Sutskever", "Geoffrey Hinton", "Jürgen Schmidhuber"]


await map(rs, "What is the most important research paper/system by {}? Reply with just the short-form of the contribution, e.g., 'Attention' or 'AlexNet'")


await map(rs, "{}'s most important research contribution")

await sorted(rs, by="impact on AI", reverse=True)


await max(rs, by="underrated")