"""Production relationships (design 6.1, 16.3).

The telemetry does not describe recipes, so a small table of *vanilla* recipes
is used and every conclusion built on it is labelled as an assumption.  The
table can be replaced via `recipes_file` (JSON: {"item": ["input", ...]}).
"""
from __future__ import annotations

import json

DEFAULT_RECIPES: dict[str, list[str]] = {
    "iron-plate": ["iron-ore"],
    "copper-plate": ["copper-ore"],
    "steel-plate": ["iron-plate"],
    "stone-brick": ["stone"],
    "iron-gear-wheel": ["iron-plate"],
    "pipe": ["iron-plate"],
    "copper-cable": ["copper-plate"],
    "electronic-circuit": ["iron-plate", "copper-cable"],
    "advanced-circuit": ["electronic-circuit", "plastic-bar", "copper-cable"],
    "processing-unit": ["electronic-circuit", "advanced-circuit", "sulfuric-acid"],
    "plastic-bar": ["coal", "petroleum-gas"],
    "sulfur": ["petroleum-gas", "water"],
    "engine-unit": ["steel-plate", "iron-gear-wheel", "pipe"],
    "inserter": ["electronic-circuit", "iron-gear-wheel", "iron-plate"],
    "transport-belt": ["iron-plate", "iron-gear-wheel"],
    "automation-science-pack": ["copper-plate", "iron-gear-wheel"],
    "logistic-science-pack": ["inserter", "transport-belt"],
    "chemical-science-pack": ["engine-unit", "advanced-circuit", "sulfur"],
}


def normalize_item(text: str) -> str:
    return "-".join(text.strip().lower().replace("_", "-").split())


class RecipeGraph:
    def __init__(self, recipes: dict[str, list[str]] | None = None):
        self.recipes = {normalize_item(k): [normalize_item(i) for i in v]
                        for k, v in (recipes if recipes is not None else DEFAULT_RECIPES).items()}
        self._consumers: dict[str, list[str]] = {}
        for product, inputs in self.recipes.items():
            for inp in inputs:
                self._consumers.setdefault(inp, []).append(product)

    @classmethod
    def load(cls, path: str = "") -> "RecipeGraph":
        if not path:
            return cls()
        with open(path) as fh:
            return cls(json.load(fh))

    def inputs(self, item: str) -> list[str]:
        return list(self.recipes.get(item, []))

    def consumers(self, item: str) -> list[str]:
        return list(self._consumers.get(item, []))

    def upstream(self, item: str, max_depth: int = 3) -> list[tuple[str, int]]:
        """Breadth-first (item, depth) pairs of everything feeding `item`."""
        seen, order, frontier = {item}, [], [item]
        for depth in range(1, max_depth + 1):
            nxt = []
            for node in frontier:
                for inp in self.inputs(node):
                    if inp not in seen:
                        seen.add(inp)
                        order.append((inp, depth))
                        nxt.append(inp)
            frontier = nxt
        return order

    def downstream(self, item: str, max_depth: int = 3) -> list[tuple[str, int]]:
        """Breadth-first (item, depth) pairs of everything that (transitively) consumes `item`."""
        seen, order, frontier = {item}, [], [item]
        for depth in range(1, max_depth + 1):
            nxt = []
            for node in frontier:
                for c in self.consumers(node):
                    if c not in seen:
                        seen.add(c)
                        order.append((c, depth))
                        nxt.append(c)
            frontier = nxt
        return order

    def known_items(self) -> set[str]:
        return set(self.recipes) | set(self._consumers)
