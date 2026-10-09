"""Complete source-matched native candidates, without labels in student input."""

from .table_pilot import parse_table


def native_repair_inputs(predictions, sources, *, role):
    if role != "model_dev":
        raise ValueError("This diagnostic allows only model-dev sources")
    expected = {r["family_id"]: r for r in sources}
    actual = {r["family_id"]: r for r in predictions}
    if (
        not expected
        or len(expected) != len(sources)
        or len(actual) != len(predictions)
        or expected.keys() != actual.keys()
    ):
        raise ValueError("Complete unique native source/candidate coverage required")
    rows = []
    for source in sources:
        candidate = actual[source["family_id"]]
        if (
            any(candidate.get(k) != v for k, v in source.items())
            or candidate["parser_error"]
            or not isinstance(candidate["prediction"], str)
        ):
            raise ValueError("Native identity differs or parser failure needs a separate protocol")
        parse_table(candidate["prediction"])
        rows.append(
            {
                "sample_id": source["family_id"] + "-native",
                **source,
                "prediction": candidate["prediction"],
            }
        )
    return rows
