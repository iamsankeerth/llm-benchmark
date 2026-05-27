"""
Test Progress Tracker - Checkpoint/Resume System for Model Testing

Provides functionality to track progress of each model test, save checkpoints,
and resume interrupted testing sessions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.artifact_store import BenchmarkArtifactStore

class TestTracker:
    def __init__(self, progress_file: str = None):
        if progress_file is None:
            base_dir = Path(__file__).parent.parent
            self.progress_file = base_dir / "test_progress.json"
        else:
            self.progress_file = Path(progress_file)
        self.store = BenchmarkArtifactStore(self.progress_file.parent)
        self.store.progress_file = self.progress_file
        
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_progress_file()
    
    def _ensure_progress_file(self):
        """Initialize progress file if it doesn't exist."""
        if not self.progress_file.exists():
            self._save({"last_updated": None, "models": {}})
    
    def _load(self) -> dict:
        """Load progress data from file."""
        data = self.store.load_progress()
        data.setdefault("last_updated", None)
        data.setdefault("models", {})
        return data
    
    def _save(self, data: dict):
        """Save progress data to file."""
        self.store.save_progress(data)
    
    def init_model(self, model_name: str, category: str, total_prompts: int = 50,
                   model_metadata: dict = None):
        """Initialize tracking for a new model.

        model_name is used as the tracker key (should be queue_id).
        model_metadata contains identity fields stored in the record.
        """
        data = self._load()
        
        if model_name not in data["models"]:
            entry = {
                "model_name": model_metadata.get("requested_name", model_name) if model_metadata else model_name,
                "category": category,
                "prompts_completed": 0,
                "total_prompts": total_prompts,
                "started_at": None,
                "completed_at": None,
                "csv_file": None,
                "error": None,
                "last_checkpoint": None,
                "status": "pending",
            }
            if model_metadata:
                entry.update({
                    "queue_id": model_metadata.get("queue_id", model_name),
                    "requested_name": model_metadata.get("requested_name", ""),
                    "source": model_metadata.get("source", ""),
                    "resolved_runtime": model_metadata.get("resolved_runtime", ""),
                    "resolved_model_ref": model_metadata.get("resolved_model_ref", ""),
                    "ollama_tag": model_metadata.get("ollama_tag", ""),
                    "hf_repo": model_metadata.get("hf_repo", ""),
                    "variant_note": model_metadata.get("variant_note", ""),
                    "fit_level": model_metadata.get("fit_level", ""),
                    "size": model_metadata.get("size", ""),
                    "estimated_tps": model_metadata.get("estimated_tps", 0),
                    "is_moe": model_metadata.get("is_moe", False),
                })
            data["models"][model_name] = entry
            self._save(data)
    
    def start_model(self, model_name: str):
        """Mark a model as in progress."""
        data = self._load()
        
        if model_name in data["models"]:
            data["models"][model_name]["status"] = "in_progress"
            data["models"][model_name]["started_at"] = datetime.now().isoformat()
            self._save(data)
    
    def update_checkpoint(self, model_name: str, prompts_completed: int, csv_file: str = None):
        """Update checkpoint after each prompt."""
        data = self._load()
        
        if model_name in data["models"]:
            data["models"][model_name]["prompts_completed"] = prompts_completed
            data["models"][model_name]["last_checkpoint"] = datetime.now().isoformat()
            if csv_file:
                data["models"][model_name]["csv_file"] = csv_file
            self._save(data)
    
    def complete_model(self, model_name: str, csv_file: str):
        """Mark a model as completed."""
        data = self._load()
        
        if model_name in data["models"]:
            total = data["models"][model_name].get("total_prompts", 50)
            data["models"][model_name]["status"] = "completed"
            data["models"][model_name]["prompts_completed"] = total
            data["models"][model_name]["completed_at"] = datetime.now().isoformat()
            data["models"][model_name]["csv_file"] = csv_file
            self._save(data)
    
    def fail_model(self, model_name: str, error: str):
        """Mark a model as failed with error reason."""
        data = self._load()
        
        if model_name in data["models"]:
            data["models"][model_name]["status"] = "failed"
            data["models"][model_name]["error"] = error
            data["models"][model_name]["completed_at"] = datetime.now().isoformat()
            self._save(data)

    def skip_model(self, model_name: str, reason: str, skip_status: str = "skipped"):
        """Mark a model as skipped with a reason (e.g. provider_unsupported, deferred_vision)."""
        data = self._load()

        if model_name in data["models"]:
            data["models"][model_name]["status"] = skip_status
            data["models"][model_name]["error"] = reason
            data["models"][model_name]["completed_at"] = datetime.now().isoformat()
            self._save(data)
    
    def is_model_complete(self, model_name: str) -> bool:
        """Check if a model has been successfully completed."""
        data = self._load()
        return data["models"].get(model_name, {}).get("status") == "completed"
    
    def is_model_failed(self, model_name: str) -> tuple[bool, Optional[str]]:
        """Check if a model has failed and return error message."""
        data = self._load()
        model_data = data["models"].get(model_name, {})
        if model_data.get("status") == "failed":
            return True, model_data.get("error")
        return False, None
    
    def get_model_progress(self, model_name: str) -> tuple[str, int, int]:
        """
        Get current progress for a model.
        Returns: (status, prompts_completed, total_prompts)
        """
        data = self._load()
        model_data = data["models"].get(model_name, {})
        status = model_data.get("status", "pending")
        completed = model_data.get("prompts_completed", 0)
        total = model_data.get("total_prompts", 50)
        return status, completed, total
    
    def get_pending_models(self) -> list[dict]:
        """Get list of models that haven't been completed or failed."""
        data = self._load()
        pending = []
        for model_name, model_data in data["models"].items():
            if model_data.get("status") in ["pending", "in_progress"]:
                pending.append({
                    "name": model_name,
                    "category": model_data.get("category"),
                    "status": model_data.get("status"),
                    "prompts_completed": model_data.get("prompts_completed", 0),
                    "total_prompts": model_data.get("total_prompts", 50)
                })
        return pending
    
    def get_all_progress(self) -> dict:
        """Get all progress data."""
        return self._load()
    
    def generate_status_report(self) -> str:
        """Generate a human-readable status report with Time Spent and ETA calculations."""
        data = self._load()
        models = data.get("models", {})
        
        total = len(models)
        completed = sum(1 for m in models.values() if m.get("status") == "completed")
        in_progress = sum(1 for m in models.values() if m.get("status") == "in_progress")
        failed = sum(1 for m in models.values() if m.get("status") == "failed")
        skipped = sum(1 for m in models.values() if m.get("status") in ("skipped", "provider_unsupported", "deferred_vision"))
        pending = sum(1 for m in models.values() if m.get("status") == "pending")
        
        # Calculate Global ETA
        global_completed_pts = 0
        global_elapsed_sec = 0.0
        
        for m_data in models.values():
            status = m_data.get("status")
            if not m_data.get("started_at"):
                continue
                
            start = datetime.fromisoformat(m_data["started_at"])
            
            if status == "completed" and m_data.get("completed_at"):
                end = datetime.fromisoformat(m_data["completed_at"])
                global_elapsed_sec += (end - start).total_seconds()
                global_completed_pts += m_data.get("total_prompts", 50)
            elif status in ["in_progress", "failed"]:
                now = datetime.fromisoformat(m_data.get("last_checkpoint", datetime.now().isoformat()))
                global_elapsed_sec += (now - start).total_seconds()
                global_completed_pts += m_data.get("prompts_completed", 0)

        global_eta_str = "Calculating..."
        if global_completed_pts > 0:
            avg_sec_per_pt = global_elapsed_sec / global_completed_pts
            global_total_pts = sum(m.get("total_prompts", 50) for m in models.values())
            pts_remaining = global_total_pts - global_completed_pts
            eta_sec = avg_sec_per_pt * pts_remaining
            
            hours = int(eta_sec // 3600)
            minutes = int((eta_sec % 3600) // 60)
            global_eta_str = f"~{hours}h {minutes}m" if eta_sec > 0 else "Finishing..."
        
        report = f"""# Test Progress Dashboard
**Last Updated**: {data.get('last_updated', 'Never')}

## Summary
- **Total Pipeline ETA**: {global_eta_str} ({pending + in_progress} models remaining)
- **Total Models**: {total}
- **Completed**: {completed}
- **In Progress**: {in_progress}
- **Failed**: {failed}
- **Skipped**: {skipped}
- **Pending**: {pending}

## Model Status by Category

"""
        categories = {}
        for model_name, model_data in models.items():
            cat = model_data.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((model_name, model_data))
        
        status_icons = {"completed": "✅", "in_progress": "🔄", "failed": "❌", "pending": "⏳", "skipped": "⏭️", "provider_unsupported": "🚫", "deferred_vision": "👁️"}
        
        for category, model_list in sorted(categories.items()):
            report += f"### {category} ({len(model_list)} models)\n\n"
            report += "| Model | Source/Runtime | Status | Progress | Time Spent | ETA | Error |\n"
            report += "|-------|---------------|--------|----------|------------|-----|-------|\n"
            
            for key, model_data in model_list:
                status = model_data.get("status", "pending")
                icon = status_icons.get(status, "❓")
                display_name = model_data.get("requested_name") or model_data.get("model_name", key)
                runtime = model_data.get("resolved_runtime") or model_data.get("source", "?")
                completed_pts = model_data.get("prompts_completed", 0)
                total_pts = model_data.get("total_prompts", 50)
                
                time_spent = "-"
                eta = "-"
                
                if model_data.get("started_at"):
                    start = datetime.fromisoformat(model_data["started_at"])
                    if status == "completed" and model_data.get("completed_at"):
                        end = datetime.fromisoformat(model_data["completed_at"])
                        time_spent = f"{(end - start).total_seconds() / 60:.1f}m"
                    elif status in ["in_progress", "failed"]:
                        now = datetime.fromisoformat(model_data.get("last_checkpoint", datetime.now().isoformat()))
                        elapsed = (now - start).total_seconds()
                        time_spent = f"{elapsed / 60:.1f}m"
                        
                        if status == "in_progress" and completed_pts > 0:
                            time_per_pt = elapsed / completed_pts
                            remaining_pts = total_pts - completed_pts
                            eta_mins = (time_per_pt * remaining_pts) / 60
                            eta = f"~{eta_mins:.1f}m" if remaining_pts > 0 else "Finishing..."
                
                if status in ("failed", "skipped", "provider_unsupported", "deferred_vision"):
                    error = (model_data.get("error") or "Unknown").replace('\n', ' ')[:40]
                    report += f"| {display_name} | {runtime} | {icon} {status} | {completed_pts}/{total_pts} | {time_spent} | - | {error}... |\n"
                else:
                    report += f"| {display_name} | {runtime} | {icon} {status.title()} | {completed_pts}/{total_pts} | {time_spent} | {eta} | - |\n"
            report += "\n"
        
        return report
    
    def save_status_to_file(self, filepath: str = None):
        """Save the status report to a markdown file."""
        if filepath is None:
            filepath = self.store.status_file
        
        report = self.generate_status_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(filepath)
    
    def generate_detailed_model_report(self, model_name: str) -> str:
        """Generate a detailed report for a specific model."""
        data = self._load()
        model_data = data.get("models", {}).get(model_name, {})
        
        if not model_data:
            return f"No data found for model: {model_name}"
        
        total_pts = model_data.get("total_prompts", 50)
        completed_pts = model_data.get("prompts_completed", 0)
        display_name = model_data.get("requested_name") or model_data.get("model_name", model_name)
        runtime = model_data.get("resolved_runtime") or model_data.get("source", "?")
        model_ref = model_data.get("resolved_model_ref") or model_data.get("ollama_tag") or model_data.get("hf_repo", "")
        variant = model_data.get("variant_note", "")
        report = f"""# Detailed Model Report: {display_name}

## Basic Information
- **Queue ID**: {model_name}
- **Category**: {model_data.get('category', 'Unknown')}
- **Status**: {model_data.get('status', 'pending').upper()}
- **Runtime**: {runtime}
- **Resolved Model Ref**: {model_ref}
- **Fit Level**: {model_data.get('fit_level', '?')}
- **Size**: {model_data.get('size', '?')}
- **Est. TPS**: {model_data.get('estimated_tps', 0)}
- **MoE**: {model_data.get('is_moe', False)}

## Progress
- **Prompts Completed**: {completed_pts} / {total_pts}
- **Completion**: {completed_pts / max(total_pts, 1) * 100:.1f}%

## Timing
- **Started At**: {model_data.get('started_at', 'Not started')}
- **Last Checkpoint**: {model_data.get('last_checkpoint', '-')}
- **Completed At**: {model_data.get('completed_at', 'Not completed')}

## Results
- **CSV File**: {model_data.get('csv_file', 'Not generated')}

"""
        if variant:
            report += f"## Variant Note\n- {variant}\n\n"
        
        if model_data.get('status') == 'failed':
            report += f"""## Error Information
- **Error**: {model_data.get('error', 'Unknown error')}

"""
        
        
        return report


def main():
    """CLI interface for test tracker."""
    tracker = TestTracker()
    
    import argparse
    parser = argparse.ArgumentParser(description="Test Progress Tracker")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument("--pending", action="store_true", help="List pending models")
    parser.add_argument("--init", metavar="MODEL", help="Initialize a model")
    parser.add_argument("--complete", metavar="MODEL", help="Mark model as complete")
    parser.add_argument("--fail", metavar="MODEL", help="Mark model as failed")
    parser.add_argument("--checkpoint", nargs=2, metavar=("MODEL", "COUNT"), help="Update checkpoint")
    
    args = parser.parse_args()
    
    if args.status:
        print(tracker.generate_status_report())
    elif args.pending:
        for m in tracker.get_pending_models():
            print(f"{m['name']} ({m['category']}) - {m['prompts_completed']}/{m.get('total_prompts', 50)} - {m['status']}")
    elif args.init:
        tracker.init_model(args.init, "Unknown")
        print(f"Initialized: {args.init}")


if __name__ == "__main__":
    main()
