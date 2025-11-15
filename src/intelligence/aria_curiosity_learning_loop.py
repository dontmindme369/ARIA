#!/usr/bin/env python3
"""
aria_curiosity_learning_loop.py - Learning Loop Integration

Connects curiosity telemetry to the bandit reward system for continuous improvement.

Features:
1. Monitors curiosity_metrics telemetry files
2. Aggregates gap scores, Socratic question quality, follow-up needs
3. Feeds enhanced reward signals to bandit state
4. Tracks curiosity-driven learning metrics
5. Identifies patterns in successful curiosity-driven retrievals

Usage:
    # Run as background service
    python aria_curiosity_learning_loop.py --telemetry-dir var/telemetry

    # One-time batch processing
    python aria_curiosity_learning_loop.py --batch --telemetry-dir var/telemetry
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import sys


class CuriosityLearningLoop:
    """
    Learns from curiosity-driven retrieval patterns to improve future retrievals.
    
    Reward Components:
    - Base retrieval quality (coverage, exemplar fit, etc.)
    - Gap detection accuracy (penalize unnecessary gaps, reward genuine ones)
    - Socratic question relevance (based on follow-up resolution success)
    - Follow-up effectiveness (did follow-up improve answer quality?)
    """
    
    def __init__(
        self,
        telemetry_dir: Path,
        state_path: Path,
        batch_size: int = 100,
        learning_rate: float = 0.1
    ):
        self.telemetry_dir = telemetry_dir
        self.state_path = state_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Curiosity learning state
        self.curiosity_state = self._load_curiosity_state()
        
        # Track processed files
        self.processed_files: set[str] = set(self.curiosity_state.get('processed_files', []))
        
        # Metrics aggregation
        self.metrics_buffer: List[Dict[str, Any]] = []
    
    def _load_curiosity_state(self) -> Dict[str, Any]:
        """Load existing curiosity learning state"""
        state_file = self.state_path.parent / 'curiosity_learning_state.json'
        
        if state_file.exists():
            try:
                return json.loads(state_file.read_text())
            except:
                pass
        
        return {
            'version': '1.0',
            'processed_files': [],
            'total_processed': 0,
            'gap_detection_stats': {
                'true_positives': 0,  # Gaps that led to useful follow-ups
                'false_positives': 0,  # Unnecessary gap detections
                'precision': 0.0
            },
            'socratic_effectiveness': {
                'questions_generated': 0,
                'questions_resolved': 0,
                'avg_resolution_quality': 0.0
            },
            'preset_curiosity_scores': {},  # Track which presets trigger good curiosity
            'gap_patterns': {},  # Learn which gap types are most valuable
            'last_update': datetime.now().isoformat()
        }
    
    def _save_curiosity_state(self):
        """Save curiosity learning state"""
        state_file = self.state_path.parent / 'curiosity_learning_state.json'
        state_file.write_text(json.dumps(self.curiosity_state, indent=2))
    
    def scan_new_telemetry(self) -> List[Dict[str, Any]]:
        """Scan for new curiosity_metrics telemetry files"""
        new_metrics = []
        
        if not self.telemetry_dir.exists():
            return new_metrics
        
        for file in self.telemetry_dir.glob('curiosity_metrics_*.txt'):
            if file.name in self.processed_files:
                continue
            
            try:
                content = file.read_text()
                metrics = json.loads(content)
                metrics['_file'] = file.name
                metrics['_timestamp'] = file.stat().st_mtime
                new_metrics.append(metrics)
                self.processed_files.add(file.name)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        return new_metrics
    
    def compute_curiosity_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Compute reward signal from curiosity metrics
        
        Reward factors:
        - Gap detection accuracy (0.0-1.0)
        - Socratic question quality (0.0-1.0)
        - Follow-up necessity (0.0-1.0)
        - Curiosity level calibration (0.0-1.0)
        
        Returns reward in range [-0.2, +0.2] to add to base reward
        """
        reward = 0.0
        
        # 1. Gap Detection Accuracy (40% weight)
        gap_score = metrics.get('gap_score', 0.0)
        has_gaps = metrics.get('has_gaps', False)
        needs_followup = metrics.get('needs_followup', False)
        
        if has_gaps:
            # Reward genuine gaps (gap_score > 0.4)
            if gap_score > 0.4:
                reward += 0.08 * min(gap_score, 1.0)
            else:
                # Small penalty for unnecessary gap detection
                reward -= 0.02
        else:
            # Small reward for clean retrievals (no gaps)
            reward += 0.04
        
        # 2. Socratic Question Quality (30% weight)
        socratic_count = metrics.get('socratic_questions', 0)
        
        if socratic_count > 0:
            # Reward appropriate number of questions (1-3 optimal)
            if 1 <= socratic_count <= 3:
                reward += 0.06
            elif socratic_count > 3:
                # Slight penalty for too many questions
                reward -= 0.01
        
        # 3. Follow-up Calibration (20% weight)
        if needs_followup:
            # If gap score justifies follow-up (> 0.5), reward
            if gap_score > 0.5:
                reward += 0.04
            else:
                # Penalty for unnecessary follow-up triggers
                reward -= 0.03
        
        # 4. Curiosity Level Calibration (10% weight)
        curiosity_level = metrics.get('curiosity_level', 0.0)
        
        # Reward moderate curiosity (0.4-0.7 range is optimal)
        if 0.4 <= curiosity_level <= 0.7:
            reward += 0.02
        elif curiosity_level > 0.8:
            # High curiosity might indicate retrieval quality issues
            reward -= 0.01
        
        return max(-0.2, min(0.2, reward))
    
    def update_bandit_state(self, metrics: Dict[str, Any], curiosity_reward: float):
        """
        Update bandit state with curiosity-enhanced reward
        
        This integrates with existing telemetry logs by:
        1. Finding corresponding query in telemetry log
        2. Extracting preset used for that query
        3. Adding curiosity_reward to that preset's reward tracking
        """
        query_id = metrics.get('query_id', '')
        query = metrics.get('query', '')
        
        # Load bandit state
        if not self.state_path.exists():
            print(f"Warning: Bandit state not found at {self.state_path}")
            return
        
        try:
            state = json.loads(self.state_path.read_text())
        except Exception as e:
            print(f"Error loading bandit state: {e}")
            return
        
        # Find matching query in telemetry log
        telemetry_log = self.telemetry_dir / 'telemetry.jsonl'
        if not telemetry_log.exists():
            return
        
        preset_name = None
        base_reward = None
        
        # Search for matching query (recent entries only)
        try:
            lines = telemetry_log.read_text().strip().split('\n')
            for line in reversed(lines[-100:]):  # Check last 100 entries
                try:
                    entry = json.loads(line)
                    if entry.get('query', '').strip() == query.strip():
                        preset_name = entry.get('preset_name')
                        base_reward = entry.get('reward')
                        break
                except:
                    continue
        except Exception as e:
            print(f"Error reading telemetry log: {e}")
            return
        
        if not preset_name:
            # Fallback: use global default or most recent preset
            presets = state.get('presets', {})
            if presets:
                # Use preset with highest recent usage
                preset_name = max(presets.keys(), key=lambda k: presets[k].get('n', 0))
        
        if not preset_name:
            return
        
        # Update preset with curiosity reward
        if 'presets' not in state:
            state['presets'] = {}
        
        if preset_name not in state['presets']:
            state['presets'][preset_name] = {
                'n': 0,
                'sum_reward': 0.0,
                'sum_sq_reward': 0.0,
                'curiosity_enhanced': True
            }
        
        preset_state = state['presets'][preset_name]
        
        # Add curiosity reward to running average
        enhanced_reward = (base_reward or 0.5) + curiosity_reward
        
        preset_state['n'] += 1
        preset_state['sum_reward'] += enhanced_reward
        preset_state['sum_sq_reward'] += enhanced_reward ** 2
        preset_state['curiosity_enhanced'] = True
        preset_state['last_curiosity_reward'] = curiosity_reward
        preset_state['last_update'] = datetime.now().isoformat()
        
        # Track curiosity-specific stats
        if preset_name not in self.curiosity_state['preset_curiosity_scores']:
            self.curiosity_state['preset_curiosity_scores'][preset_name] = {
                'total_queries': 0,
                'avg_gap_score': 0.0,
                'avg_curiosity_reward': 0.0,
                'followup_rate': 0.0
            }
        
        pcs = self.curiosity_state['preset_curiosity_scores'][preset_name]
        pcs['total_queries'] += 1
        pcs['avg_gap_score'] = (
            (pcs['avg_gap_score'] * (pcs['total_queries'] - 1) + metrics.get('gap_score', 0)) 
            / pcs['total_queries']
        )
        pcs['avg_curiosity_reward'] = (
            (pcs['avg_curiosity_reward'] * (pcs['total_queries'] - 1) + curiosity_reward)
            / pcs['total_queries']
        )
        if metrics.get('needs_followup'):
            pcs['followup_rate'] = (
                (pcs['followup_rate'] * (pcs['total_queries'] - 1) + 1.0)
                / pcs['total_queries']
            )
        
        # Save updated state
        self.state_path.write_text(json.dumps(state, indent=2))
        
        print(f"✓ Updated {preset_name}: curiosity_reward={curiosity_reward:+.3f}, "
              f"enhanced_reward={enhanced_reward:.3f}")
    
    def update_gap_patterns(self, metrics: Dict[str, Any]):
        """Learn which types of gaps are most valuable"""
        gap_score = metrics.get('gap_score', 0.0)
        needs_followup = metrics.get('needs_followup', False)
        
        # Track gap score distribution
        gap_bucket = f"{int(gap_score * 10) / 10:.1f}"
        
        if 'gap_patterns' not in self.curiosity_state:
            self.curiosity_state['gap_patterns'] = {}
        
        if gap_bucket not in self.curiosity_state['gap_patterns']:
            self.curiosity_state['gap_patterns'][gap_bucket] = {
                'count': 0,
                'followup_rate': 0.0
            }
        
        pattern = self.curiosity_state['gap_patterns'][gap_bucket]
        pattern['count'] += 1
        
        if needs_followup:
            pattern['followup_rate'] = (
                (pattern['followup_rate'] * (pattern['count'] - 1) + 1.0)
                / pattern['count']
            )
    
    def process_metrics(self, metrics: Dict[str, Any]):
        """Process a single curiosity metrics entry"""
        # Compute curiosity reward
        curiosity_reward = self.compute_curiosity_reward(metrics)
        
        # Update bandit state with curiosity reward
        self.update_bandit_state(metrics, curiosity_reward)
        
        # Learn gap patterns
        self.update_gap_patterns(metrics)
        
        # Update global stats
        self.curiosity_state['total_processed'] += 1
        
        if metrics.get('socratic_questions', 0) > 0:
            stats = self.curiosity_state['socratic_effectiveness']
            stats['questions_generated'] += metrics['socratic_questions']
    
    def process_batch(self):
        """Process a batch of new telemetry"""
        new_metrics = self.scan_new_telemetry()
        
        if not new_metrics:
            return 0
        
        print(f"\nProcessing {len(new_metrics)} new curiosity metrics...")
        
        for metrics in new_metrics:
            try:
                self.process_metrics(metrics)
            except Exception as e:
                print(f"Error processing metrics: {e}")
        
        # Update state
        self.curiosity_state['processed_files'] = list(self.processed_files)
        self.curiosity_state['last_update'] = datetime.now().isoformat()
        self._save_curiosity_state()
        
        print(f"✓ Processed {len(new_metrics)} metrics")
        print(f"  Total processed: {self.curiosity_state['total_processed']}")
        print(f"  Presets tracked: {len(self.curiosity_state['preset_curiosity_scores'])}")
        
        return len(new_metrics)
    
    def run_continuous(self, interval: int = 10):
        """Run continuous learning loop"""
        print("Starting continuous curiosity learning loop...")
        print(f"  Telemetry dir: {self.telemetry_dir}")
        print(f"  Bandit state: {self.state_path}")
        print(f"  Scan interval: {interval}s")
        print()
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}")
                
                count = self.process_batch()
                
                if count > 0:
                    print(f"  ✓ Processed {count} new metrics")
                else:
                    print(f"  No new metrics")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nShutting down learning loop...")
            self._save_curiosity_state()
            print("✓ State saved")


def main():
    parser = argparse.ArgumentParser(
        description='ARIA Curiosity Learning Loop - Integrates curiosity telemetry with bandit learning'
    )
    parser.add_argument(
        '--telemetry-dir',
        type=Path,
        default=Path('var/telemetry'),
        help='Directory containing curiosity_metrics telemetry files'
    )
    parser.add_argument(
        '--state-path',
        type=Path,
        default=Path.home() / '.rag_bandit_state.json',
        help='Path to bandit state file'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run once in batch mode instead of continuous'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Scan interval in seconds (continuous mode)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.telemetry_dir.exists():
        print(f"Error: Telemetry directory not found: {args.telemetry_dir}")
        return 1
    
    # Create learning loop
    loop = CuriosityLearningLoop(
        telemetry_dir=args.telemetry_dir,
        state_path=args.state_path
    )
    
    if args.batch:
        # Batch mode
        print("Running in batch mode...")
        count = loop.process_batch()
        print(f"\nProcessed {count} metrics")
        return 0
    else:
        # Continuous mode
        loop.run_continuous(interval=args.interval)
        return 0


if __name__ == '__main__':
    sys.exit(main())
