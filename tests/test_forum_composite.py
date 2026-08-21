"""Validate the forum composite wiring against the REAL Finding model + the
leaderboard sort logic. Uses an in-memory SQLite via a minimal Flask app (no
boto3/runpod — we don't import app.py). Run with the throwaway venv:
    /tmp/forumtest/bin/python tests/test_forum_composite.py
"""
import json
from flask import Flask
from aar.web_ui.backend.models import db, Finding


def _rank(x):
    # Mirror of app.py /api/leaderboard sort key.
    if x.get("headline_pct") is not None:
        return x["headline_pct"]
    return x["pgr"] * 100 if x.get("pgr") is not None else -1e9


def main():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    db.init_app(app)
    with app.app_context():
        db.create_all()

        # A multi-benchmark finding (new path) + a legacy PGR finding.
        comp = {"headline_pct": 12.3, "passes_filter": True,
                "closed_pct": {"sycophancy_eval": 20.0, "syc_eval": 4.6}}
        db.session.add(Finding(
            post_id="p1", title="antisyc_v1", content="x", finding_type="result",
            suite="sycophancy", headline_pct=12.3, composite_scores=json.dumps(comp),
        ))
        db.session.add(Finding(
            post_id="p2", title="legacy_w2s", content="y", finding_type="result", pgr=0.30,
        ))
        db.session.commit()

        rows = [f.to_dict() for f in Finding.query.all()]
        # 1) composite round-trips through the model
        f1 = next(r for r in rows if r["post_id"] == "p1")
        assert f1["suite"] == "sycophancy", f1["suite"]
        assert f1["headline_pct"] == 12.3
        assert f1["composite_scores"]["closed_pct"]["sycophancy_eval"] == 20.0, f1["composite_scores"]
        # 2) legacy finding still works (composite None)
        f2 = next(r for r in rows if r["post_id"] == "p2")
        assert f2["composite_scores"] is None and f2["pgr"] == 0.30
        # 3) leaderboard sort: composite headline (12.3) outranks pgr 0.30 (->30) ?
        #    pgr 0.30 -> 30 > 12.3, so legacy ranks higher here — verify ordering is by the rule
        ordered = sorted(rows, key=_rank, reverse=True)
        assert _rank(f2) == 30.0 and _rank(f1) == 12.3
        assert ordered[0]["post_id"] == "p2"  # 30 > 12.3
        # 4) a higher-headline finding outranks both
        f3 = {"headline_pct": 55.0, "pgr": None, "post_id": "p3"}
        assert _rank(f3) == 55.0 and max((f1, f2, f3), key=_rank)["post_id"] == "p3"

    print("OK: forum composite wiring validated (model round-trip + sort)")


if __name__ == "__main__":
    main()
