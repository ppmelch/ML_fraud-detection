"""
Reconciliation between dataset state names and the provided India GeoJSON.

The frontend colours a polygon by looking up a metric under the polygon's own
name. A name the backend spells differently is not an error the browser can
report — the polygon simply stays grey. So the join is made explicit here,
validated, and every unmatched name on either side is carried into the
payload instead of being dropped.
"""

import json
from pathlib import Path

from backend.src.modeling.config import (
    GEOJSON_ID_PROPERTY,
    GEOJSON_NAME_PROPERTY,
    GEOJSON_PATH,
    STATE_NAME_MAP,
)


class StateGeoMapper:
    """
    Match dataset state names to GeoJSON features and validate the join.

    Attributes
    ----------
    geojson_path : Path
        Location of the India GeoJSON supplied with the project.
    name_map : dict[str, str]
        Explicit dataset-spelling to GeoJSON-spelling overrides.
    features : list[dict]
        Feature objects read from the GeoJSON.
    """

    def __init__(
        self,
        geojson_path: Path = GEOJSON_PATH,
        name_map: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the mapper and read the GeoJSON.

        Parameters
        ----------
        geojson_path : Path, optional
            Path to the GeoJSON file. Defaults to ``GEOJSON_PATH``.
        name_map : dict[str, str], optional
            Dataset-to-GeoJSON name overrides. Defaults to ``STATE_NAME_MAP``.

        Raises
        ------
        FileNotFoundError
            If the GeoJSON file is absent.
        ValueError
            If the file is not a FeatureCollection, or if two features share
            the same name, which would make the join ambiguous.
        """
        self.geojson_path = Path(geojson_path)
        self.name_map = dict(name_map if name_map is not None else STATE_NAME_MAP)

        if not self.geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON not found: {self.geojson_path}")

        with self.geojson_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("type") != "FeatureCollection":
            raise ValueError(
                f"Expected a FeatureCollection, got {payload.get('type')!r}"
            )

        self.features = payload["features"]

        names = [
            feature["properties"][GEOJSON_NAME_PROPERTY]
            for feature in self.features
        ]

        duplicates = sorted({n for n in names if names.count(n) > 1})

        if duplicates:
            raise ValueError(f"Duplicate names in GeoJSON: {duplicates}")

        self._by_name = {
            feature["properties"][GEOJSON_NAME_PROPERTY]: feature["properties"]
            for feature in self.features
        }

    @property
    def geojson_names(self) -> list[str]:
        """
        Names of every feature in the GeoJSON.

        Returns
        -------
        list[str]
            Sorted feature names.
        """
        return sorted(self._by_name)

    def to_geojson_name(self, dataset_state: str) -> str | None:
        """
        Translate a dataset state name into its GeoJSON name.

        Parameters
        ----------
        dataset_state : str
            State name as spelled in the transaction dataset.

        Returns
        -------
        str or None
            The matching GeoJSON name, or ``None`` when no feature matches.
        """
        candidate = self.name_map.get(dataset_state, dataset_state)

        return candidate if candidate in self._by_name else None

    def geojson_id(self, geojson_name: str) -> str | None:
        """
        Look up the stable feature identifier for a GeoJSON name.

        Parameters
        ----------
        geojson_name : str
            Name of a GeoJSON feature.

        Returns
        -------
        str or None
            The feature's ``id`` property, or ``None`` if unknown.
        """
        properties = self._by_name.get(geojson_name)

        if properties is None:
            return None

        value = properties.get(GEOJSON_ID_PROPERTY)

        return None if value is None else str(value)

    def attach(self, state_records: list[dict]) -> list[dict]:
        """
        Add the GeoJSON join keys to each state metric record.

        Parameters
        ----------
        state_records : list[dict]
            Records produced by
            :meth:`~backend.src.metrics.state_analytics.StateAnalytics.compute`,
            each carrying a ``state`` key.

        Returns
        -------
        list[dict]
            The same records with ``geojson_name``, ``geojson_id`` and
            ``matched`` added. Unmatched states keep all of their metrics and
            are marked ``matched: False`` rather than being discarded.
        """
        enriched = []

        for record in state_records:
            geojson_name = self.to_geojson_name(record["state"])

            enriched.append(
                {
                    **record,
                    "geojson_name": geojson_name,
                    "geojson_id": (
                        self.geojson_id(geojson_name) if geojson_name else None
                    ),
                    "matched": geojson_name is not None,
                }
            )

        return enriched

    def validate(self, state_records: list[dict]) -> dict:
        """
        Report the health of the dataset-to-GeoJSON join.

        Parameters
        ----------
        state_records : list[dict]
            Records carrying a ``state`` key.

        Returns
        -------
        dict
            ``matched_states``, ``unmatched_dataset_states`` (present in the
            data, absent from the map), ``geojson_states_without_data``
            (drawn on the map with no transactions), ``duplicate_states``
            (a state appearing twice in the records), the explicit renames
            actually used, and an ``ok`` flag that is True only when every
            dataset state found a polygon and no state is duplicated.
        """
        seen = [record["state"] for record in state_records]

        duplicates = sorted({s for s in seen if seen.count(s) > 1})

        matched: dict[str, str] = {}
        unmatched: list[str] = []

        for state in seen:
            geojson_name = self.to_geojson_name(state)

            if geojson_name is None:
                unmatched.append(state)
            else:
                matched[state] = geojson_name

        without_data = sorted(set(self._by_name) - set(matched.values()))

        renames_used = {
            dataset: geo
            for dataset, geo in matched.items()
            if dataset != geo
        }

        return {
            "geojson_path": str(self.geojson_path),
            "geojson_features": len(self.features),
            "dataset_states": len(set(seen)),
            "matched_states": len(matched),
            "unmatched_dataset_states": sorted(set(unmatched)),
            "geojson_states_without_data": without_data,
            "duplicate_states": duplicates,
            "renames_applied": renames_used,
            "ok": not unmatched and not duplicates,
        }
