"""Database models for experiment tracking."""
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

from aar.web_ui.backend import config

db = SQLAlchemy()


def _safe_datetime_subtract(dt1, dt2):
    """
    Safely subtract two datetimes, handling both timezone-aware and naive datetimes.
    Assumes naive datetimes are in UTC.
    """
    from datetime import timezone
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=timezone.utc)
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=timezone.utc)
    return (dt1 - dt2).total_seconds()


class Idea(db.Model):
    """Stores all research ideas with their full content.
    
    This is the single source of truth for idea metadata.
    Experiments reference ideas by name.
    """
    
    __tablename__ = 'ideas'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True, index=True)
    uid = db.Column(db.String(100), nullable=True, unique=True, index=True)  # Unique identifier for S3 paths
    
    # Full idea content
    idea_json = db.Column(db.Text, nullable=True)  # Full JSON (~4KB each)

    # Extracted fields for easy querying
    description = db.Column(db.Text, nullable=True)
    
    # Metadata
    source = db.Column(db.String(50), nullable=True)  # 'manual'
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    
    # Tags for categorization
    is_baseline = db.Column(db.Boolean, default=False, nullable=False)  # Human-verified baseline (vanilla_w2s, etc.)

    @classmethod
    def from_dict(cls, idea_dict: dict, source: str = None, is_baseline: bool = False) -> 'Idea':
        """Create an Idea from a dictionary."""
        import json
        return cls(
            name=idea_dict.get('Name', ''),
            uid=idea_dict.get('uid'),
            idea_json=json.dumps(idea_dict),
            description=idea_dict.get('Description', ''),
            source=source,
            is_baseline=is_baseline,
        )
    
    def get_dict(self) -> dict:
        """Get the full idea dictionary."""
        import json
        if not self.idea_json:
            return {'Name': self.name, 'Description': self.description or ''}
        return json.loads(self.idea_json)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'uid': self.uid,  # Include UID for API responses
            'description': self.description,
            'source': self.source,
            'is_baseline': self.is_baseline,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def to_full_dict(self) -> dict:
        """Get full idea dict including parsed JSON."""
        result = self.to_dict()
        result['idea_data'] = self.get_dict()
        return result


class Experiment(db.Model):
    """Tracks experiment execution and results.
    
    Each run of an idea+config creates a new row. Multiple runs of the same
    idea+config are allowed (historical runs are preserved).
    """

    __tablename__ = 'experiments'
    
    # No unique constraint - each run creates a new row
    # This allows tracking historical runs of the same idea+config

    id = db.Column(db.Integer, primary_key=True)
    idea_name = db.Column(db.String(200), nullable=False, index=True)
    idea_title = db.Column(db.String(500), nullable=True)
    idea_description = db.Column(db.Text, nullable=True)
    
    # Full configuration - allows same idea to be run with different configs
    dataset = db.Column(db.String(200), nullable=False, default="math-claudefilter-imbalance", index=True)
    weak_model = db.Column(db.String(200), nullable=False, default="Qwen/Qwen1.5-0.5B", index=True)
    strong_model = db.Column(db.String(200), nullable=False, default="Qwen/Qwen3-4B-Base", index=True)

    # Generalized multi-benchmark harness (additive; nullable so legacy rows are fine):
    # which suite this run targeted, and the per-benchmark composite JSON the
    # eval pod returned (headline_pct, passes_filter, closed_pct, per_benchmark).
    suite = db.Column(db.String(200), nullable=True, index=True)
    composite_scores = db.Column(db.Text, nullable=True)

    # Status: 'queued', 'running', 'completed', 'failed'
    status = db.Column(db.String(20), default='queued', nullable=False)

    # Results — DEPRECATED: leaderboard now sources from Finding table.
    # Kept for backward compat with in-flight workers and run lifecycle queries.
    pgr = db.Column(db.Float, nullable=True)
    pgr_se = db.Column(db.Float, nullable=True)
    weak_acc = db.Column(db.Float, nullable=True)
    strong_acc = db.Column(db.Float, nullable=True)
    transfer_acc = db.Column(db.Float, nullable=True)
    transfer_acc_std = db.Column(db.Float, nullable=True)
    num_seeds = db.Column(db.Integer, nullable=True)
    seeds = db.Column(db.Text, nullable=True)

    # Timing
    queue_time = db.Column(db.DateTime, default=datetime.now, nullable=False)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)

    # Logs and errors
    logs = db.Column(db.Text, nullable=True)
    error_msg = db.Column(db.Text, nullable=True)
    
    # RunPod tracking (for distributed execution)
    pod_id = db.Column(db.String(100), nullable=True)  # RunPod pod ID
    idea_uid = db.Column(db.String(100), nullable=True, index=True)  # Unique identifier for the idea (used for S3 paths)
    run_id = db.Column(db.String(50), nullable=True)  # Timestamp-based run ID for this execution (e.g., "run-20251213-143052")
    results_uploaded_to_s3 = db.Column(db.Boolean, default=False, nullable=False)  # True when results file exists in S3 (set by worker)

    # Retry tracking (for handling transient deployment failures)
    deploy_retry_count = db.Column(db.Integer, default=0, nullable=False)  # Number of deployment attempts
    last_deploy_attempt = db.Column(db.DateTime, nullable=True)  # Timestamp of last deployment attempt
    
    # Execution mode: 'local' (subprocess), 'docker' (local Docker), 'runpod' (cloud)
    execution_mode = db.Column(db.String(20), nullable=True, default=None)

    # GPU assignment for local/docker mode (e.g. "0,1,2,3")
    gpu_ids = db.Column(db.String(100), nullable=True, default=None)

    # Idea content (stored in DB to avoid file dependency)
    idea_json = db.Column(db.Text, nullable=True)  # Full idea JSON for backup
    
    # Idea metadata
    idea_created_at = db.Column(db.DateTime, nullable=True)  # When the idea was generated

    def to_dict(self):
        """Convert to dictionary for API responses."""
        import json
        duration_seconds = None
        if self.start_time and self.end_time:
            duration_seconds = _safe_datetime_subtract(self.end_time, self.start_time)
        elif self.start_time:
            # For current time, use timezone-aware datetime
            now = datetime.now()
            duration_seconds = _safe_datetime_subtract(now, self.start_time)

        # Compute S3 prefix when idea_uid and run_id exist (allows viewing logs/usage_stats before results.json)
        # Individual endpoints handle missing files gracefully
        s3_prefix = None
        if self.idea_uid and self.run_id:
            s3_prefix = (
                f"s3://{config.S3_BUCKET}/"
                f"{config.S3_IDEAS_PREFIX}{self.idea_uid}/{self.run_id}/"
            )

        # Determine if this is a baseline experiment
        BASELINE_IDEA_NAMES = {
            '_strong_ceiling', '_weak_baseline',
            'vanilla_w2s', 'critic', 'unsupervised_elicitation', 'train_only_on_confident_labels'
        }
        is_baseline = self.idea_name in BASELINE_IDEA_NAMES
        
        return {
            'id': self.id,
            'idea_name': self.idea_name,
            'idea_title': self.idea_title,
            'idea_description': self.idea_description,
            'dataset': self.dataset,
            'weak_model': self.weak_model,
            'strong_model': self.strong_model,
            'status': self.status,
            'is_baseline': is_baseline,
            'pgr': self.pgr,
            'pgr_se': self.pgr_se,
            'weak_acc': self.weak_acc,
            'strong_acc': self.strong_acc,
            'transfer_acc': self.transfer_acc,
            'transfer_acc_std': self.transfer_acc_std,
            'num_seeds': self.num_seeds,
            'seeds': json.loads(self.seeds) if self.seeds else None,
            'queue_time': self.queue_time.isoformat() if self.queue_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration_seconds,
            'logs': self.logs,
            'error_msg': self.error_msg,
            'pod_id': self.pod_id,
            'idea_uid': self.idea_uid,
            'run_id': self.run_id,
            's3_prefix': s3_prefix,
            'results_uploaded_to_s3': self.results_uploaded_to_s3,
            'deploy_retry_count': self.deploy_retry_count,
            'last_deploy_attempt': self.last_deploy_attempt.isoformat() if self.last_deploy_attempt else None,
            'execution_mode': self.execution_mode,
            'gpu_ids': self.gpu_ids,
        }



class Finding(db.Model):
    """Unified model for agent findings — merges the old Lesson + ForumPost tables.

    Created when agents share results, observations, hypotheses, etc.
    Supports voting, comments, and serves both the MCP query API and the web forum UI.
    """

    __tablename__ = 'findings'

    id = db.Column(db.Integer, primary_key=True)

    # Post identification (UUID for URL-safe lookups)
    post_id = db.Column(db.String(100), nullable=False, unique=True, index=True)

    # Content
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)  # MCP sends 'summary', stored here
    finding_type = db.Column(db.String(50), nullable=True, index=True)  # hypothesis, result, insight, error, observation

    # Source identification
    idea_uid = db.Column(db.String(100), nullable=True, index=True)
    idea_name = db.Column(db.String(200), nullable=True, index=True)
    run_id = db.Column(db.String(100), nullable=True, index=True)
    session_id = db.Column(db.String(100), nullable=True)

    # Experiment context
    dataset = db.Column(db.String(200), nullable=True, index=True)
    weak_model = db.Column(db.String(200), nullable=True)
    strong_model = db.Column(db.String(200), nullable=True)

    # Leaderboard fields
    idea_title = db.Column(db.String(500), nullable=True)  # Display title for leaderboard
    is_baseline = db.Column(db.Boolean, default=False, nullable=False)  # Marks baseline results
    seeds = db.Column(db.Text, nullable=True)  # JSON list of seed values

    # Metrics
    pgr = db.Column(db.Float, nullable=True)
    pgr_delta = db.Column(db.Float, nullable=True)
    pgr_se = db.Column(db.Float, nullable=True)
    transfer_acc = db.Column(db.Float, nullable=True)
    transfer_acc_se = db.Column(db.Float, nullable=True)
    weak_acc = db.Column(db.Float, nullable=True)
    strong_acc = db.Column(db.Float, nullable=True)
    num_seeds = db.Column(db.Integer, nullable=True)

    # Lesson-specific fields
    iteration = db.Column(db.Integer, nullable=True)
    config = db.Column(db.Text, nullable=True)  # JSON
    worked = db.Column(db.Boolean, nullable=True)

    # Snapshot / code (commit fields folded in — no separate Commit table)
    commit_id = db.Column(db.String(100), nullable=True, unique=True, index=True)
    s3_path = db.Column(db.String(500), nullable=True)
    s3_key = db.Column(db.String(500), nullable=True)
    parent_commit_id = db.Column(db.String(100), nullable=True)
    sequence_number = db.Column(db.Integer, nullable=True)
    files_snapshot = db.Column(db.Text, nullable=True)  # JSON list of files
    code_snippet = db.Column(db.Text, nullable=True)

    # Multi-benchmark harness (additive, nullable): the suite, the composite
    # headline (mean headroom-closed over safety benchmarks), and the full
    # per-benchmark composite JSON. These supersede pgr for non-W2S suites.
    suite = db.Column(db.String(200), nullable=True, index=True)
    headline_pct = db.Column(db.Float, nullable=True)
    composite_scores = db.Column(db.Text, nullable=True)  # JSON

    # Engagement
    upvotes = db.Column(db.Integer, default=0, nullable=False)
    downvotes = db.Column(db.Integer, default=0, nullable=False)
    comment_count = db.Column(db.Integer, default=0, nullable=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationship to comments
    comments = db.relationship('FindingComment', backref='finding', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self, include_comments=False):
        """Convert to dictionary for API responses."""
        config_dict = None
        if self.config:
            try:
                config_dict = json.loads(self.config)
            except json.JSONDecodeError:
                pass

        result = {
            'id': self.id,
            'post_id': self.post_id,
            'title': self.title,
            'content': self.content,
            'finding_type': self.finding_type,
            'idea_uid': self.idea_uid,
            'idea_name': self.idea_name,
            'idea_title': self.idea_title,
            'run_id': self.run_id,
            'session_id': self.session_id,
            'dataset': self.dataset,
            'suite': self.suite,
            'headline_pct': self.headline_pct,
            'composite_scores': json.loads(self.composite_scores) if self.composite_scores else None,
            'weak_model': self.weak_model,
            'strong_model': self.strong_model,
            'is_baseline': self.is_baseline,
            'pgr': self.pgr,
            'pgr_delta': self.pgr_delta,
            'pgr_se': self.pgr_se,
            'transfer_acc': self.transfer_acc,
            'transfer_acc_se': self.transfer_acc_se,
            'weak_acc': self.weak_acc,
            'strong_acc': self.strong_acc,
            'num_seeds': self.num_seeds,
            'seeds': json.loads(self.seeds) if self.seeds else None,
            'iteration': self.iteration,
            'config': config_dict,
            'worked': self.worked,
            'commit_id': self.commit_id,
            's3_path': self.s3_path,
            's3_key': self.s3_key,
            'parent_commit_id': self.parent_commit_id,
            'sequence_number': self.sequence_number,
            'files_snapshot': json.loads(self.files_snapshot) if self.files_snapshot else None,
            'file_count': len(json.loads(self.files_snapshot)) if self.files_snapshot else 0,
            'code_snippet': self.code_snippet,
            'upvotes': self.upvotes,
            'downvotes': self.downvotes,
            'score': self.upvotes - self.downvotes,
            'comment_count': self.comment_count,
            'created_at': (self.created_at.isoformat() + 'Z') if self.created_at else None,
            'updated_at': (self.updated_at.isoformat() + 'Z') if self.updated_at else None,
        }
        if include_comments:
            result['comments'] = [c.to_dict() for c in self.comments.order_by(FindingComment.created_at.asc()).all()]
        return result


class FindingComment(db.Model):
    """Comments on findings."""

    __tablename__ = 'finding_comments'

    id = db.Column(db.Integer, primary_key=True)

    # Link to finding
    finding_id = db.Column(db.Integer, db.ForeignKey('findings.id'), nullable=False, index=True)

    # Content
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=True)  # 'human' or agent session_id

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'finding_id': self.finding_id,
            'content': self.content,
            'author': self.author,
            'created_at': (self.created_at.isoformat() + 'Z') if self.created_at else None,
        }
