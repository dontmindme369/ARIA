#!/usr/bin/env python3
"""
Pattern Miner - Wave 3: Self-Learning System

Discovers successful retrieval patterns and strategy combinations.

Integration with Current ARIA:
- Analyzes aria_telemetry.py for strategy performance
- Uses bandit_context.py (contextual_bandit.py) arm selections
- Feeds auto_tuner.py with optimization insights
- Informs anchor_selector.py mode selection

Key Features:
1. Strategy combination discovery (what works together?)
2. Query pattern analysis (which queries benefit from which strategies?)
3. Temporal pattern detection (performance trends over time)
4. Context-reward correlation (what contexts predict success?)
5. Actionable insights generation

Flow:
1. Collect telemetry data with strategy selections and rewards
2. Mine patterns: "Query type X + Strategy Y → High reward"
3. Cluster successful combinations
4. Generate rules: "For queries about Z, prefer strategy W"
5. Export as tuning recommendations
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

try:
    import numpy as np
    from scipy.stats import pearsonr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    np = None  # type: ignore


class PatternMiner:
    """
    Mine successful retrieval patterns from telemetry.
    
    Patterns to discover:
    - Query features → Best strategy mapping
    - Strategy combinations that work well together
    - Anchor modes that predict high reward
    - Temporal patterns (morning vs evening, weekday vs weekend)
    - Source reliability patterns
    """
    
    def __init__(
        self,
        telemetry_dir: Path,
        min_samples: int = 20,
        confidence_threshold: float = 0.7
    ):
        self.telemetry_dir = telemetry_dir
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.patterns: List[Dict[str, Any]] = []
        self.query_features: Dict[str, List[float]] = defaultdict(list)
        self.strategy_rewards: Dict[str, List[float]] = defaultdict(list)
        self.anchor_rewards: Dict[str, List[float]] = defaultdict(list)
        
        # State
        self.state_file = Path.home() / '.aria_pattern_miner.json'
        self._load_state()
    
    def _load_state(self):
        """Load previously mined patterns"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                self.patterns = state.get('patterns', [])
            except:
                pass
    
    def _save_state(self):
        """Save mined patterns"""
        self.state_file.write_text(json.dumps({
            'patterns': self.patterns,
            'last_update': datetime.now().isoformat(),
            'total_samples': sum(len(v) for v in self.strategy_rewards.values())
        }, indent=2))
    
    def collect_data(self, max_age_days: int = 7) -> int:
        """
        Collect telemetry data for pattern mining.
        
        Args:
            max_age_days: How far back to scan
            
        Returns:
            Number of samples collected
        """
        if not self.telemetry_dir.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        samples_collected = 0
        
        for telemetry_file in self.telemetry_dir.glob('*.jsonl'):
            try:
                mtime = datetime.fromtimestamp(telemetry_file.stat().st_mtime)
                if mtime < cutoff_time:
                    continue
                
                for line in telemetry_file.read_text().split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Extract relevant fields
                        query = entry.get('query', '')
                        reward = entry.get('reward', 0.0)
                        preset = entry.get('preset', 'unknown')
                        anchor = entry.get('reasoning_mode', 'unknown')
                        
                        # Query features
                        query_len = len(query.split())
                        has_question = '?' in query
                        has_comparison = any(w in query.lower() for w in ['vs', 'versus', 'compare', 'difference'])
                        
                        # Store
                        feature_key = f"len_{min(query_len // 5, 10)}"  # Bin query lengths
                        self.query_features[feature_key].append(reward)
                        
                        self.strategy_rewards[preset].append(reward)
                        self.anchor_rewards[anchor].append(reward)
                        
                        samples_collected += 1
                    
                    except json.JSONDecodeError:
                        continue
            
            except Exception:
                continue
        
        return samples_collected
    
    def mine_strategy_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine patterns about which strategies work best.
        
        Returns:
            List of pattern dictionaries
        """
        patterns: List[Dict[str, Any]] = []
        
        # Filter strategies with enough samples
        valid_strategies = {
            strategy: rewards 
            for strategy, rewards in self.strategy_rewards.items()
            if len(rewards) >= self.min_samples
        }
        
        if not valid_strategies:
            return patterns
        
        # Overall best strategy
        strategy_means = {
            strategy: statistics.mean(rewards)
            for strategy, rewards in valid_strategies.items()
        }
        
        best_strategy = max(strategy_means.items(), key=lambda x: x[1])
        
        if best_strategy[1] >= self.confidence_threshold:
            patterns.append({
                'type': 'best_overall_strategy',
                'strategy': best_strategy[0],
                'avg_reward': best_strategy[1],
                'confidence': min(1.0, best_strategy[1] / 0.85),  # Normalize
                'samples': len(valid_strategies[best_strategy[0]]),
                'recommendation': f"Strategy '{best_strategy[0]}' consistently performs well (avg reward: {best_strategy[1]:.2f})"
            })
        
        # Identify underperforming strategies
        mean_reward = statistics.mean([r for rewards in valid_strategies.values() for r in rewards])
        
        for strategy, rewards in valid_strategies.items():
            strategy_mean = statistics.mean(rewards)
            
            if strategy_mean < mean_reward * 0.8:  # 20% below average
                patterns.append({
                    'type': 'underperforming_strategy',
                    'strategy': strategy,
                    'avg_reward': strategy_mean,
                    'baseline_reward': mean_reward,
                    'samples': len(rewards),
                    'recommendation': f"Consider tuning '{strategy}' - performing {((strategy_mean / mean_reward - 1) * 100):.1f}% below average"
                })
        
        return patterns
    
    def mine_anchor_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine patterns about anchor mode effectiveness.
        
        Returns:
            List of pattern dictionaries
        """
        patterns: List[Dict[str, Any]] = []
        
        # Filter anchors with enough samples
        valid_anchors = {
            anchor: rewards
            for anchor, rewards in self.anchor_rewards.items()
            if len(rewards) >= self.min_samples and anchor != 'unknown'
        }
        
        if not valid_anchors:
            return patterns
        
        # Anchor performance comparison
        anchor_stats = {}
        for anchor, rewards in valid_anchors.items():
            anchor_stats[anchor] = {
                'mean': statistics.mean(rewards),
                'std': statistics.stdev(rewards) if len(rewards) > 1 else 0,
                'samples': len(rewards)
            }
        
        # Find best anchor
        best_anchor = max(anchor_stats.items(), key=lambda x: x[1]['mean'])
        
        patterns.append({
            'type': 'best_anchor_mode',
            'anchor': best_anchor[0],
            'avg_reward': best_anchor[1]['mean'],
            'stability': 1.0 - min(1.0, best_anchor[1]['std']),
            'samples': best_anchor[1]['samples'],
            'recommendation': f"Anchor mode '{best_anchor[0]}' shows strong performance (avg: {best_anchor[1]['mean']:.2f})"
        })
        
        # Identify high-variance anchors
        for anchor, stats in anchor_stats.items():
            if stats['std'] > 0.25:  # High variance
                patterns.append({
                    'type': 'unstable_anchor',
                    'anchor': anchor,
                    'variance': stats['std'],
                    'samples': stats['samples'],
                    'recommendation': f"Anchor '{anchor}' has inconsistent performance (std: {stats['std']:.2f}) - investigate edge cases"
                })
        
        return patterns
    
    def mine_query_length_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine patterns about query length vs performance.
        
        Returns:
            List of pattern dictionaries
        """
        patterns: List[Dict[str, Any]] = []
        
        if not self.query_features:
            return patterns
        
        # Analyze by query length bins
        length_performance = {}
        for length_bin, rewards in self.query_features.items():
            if len(rewards) >= self.min_samples:
                length_performance[length_bin] = {
                    'mean': statistics.mean(rewards),
                    'samples': len(rewards)
                }
        
        if not length_performance:
            return patterns
        
        # Find optimal query length range
        best_length = max(length_performance.items(), key=lambda x: x[1]['mean'])
        worst_length = min(length_performance.items(), key=lambda x: x[1]['mean'])
        
        if best_length[1]['mean'] > worst_length[1]['mean'] * 1.2:  # 20% difference
            patterns.append({
                'type': 'query_length_effect',
                'optimal_length_bin': best_length[0],
                'optimal_reward': best_length[1]['mean'],
                'worst_length_bin': worst_length[0],
                'worst_reward': worst_length[1]['mean'],
                'recommendation': f"Queries in '{best_length[0]}' range perform {((best_length[1]['mean'] / worst_length[1]['mean'] - 1) * 100):.1f}% better"
            })
        
        return patterns
    
    def mine_temporal_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine temporal patterns (time-of-day, day-of-week effects).
        
        Note: Requires timestamp data in telemetry
        
        Returns:
            List of pattern dictionaries
        """
        patterns: List[Dict[str, Any]] = []
        
        # This would require parsing timestamps from telemetry
        # Placeholder for future implementation
        
        return patterns
    
    def mine_all_patterns(self, max_age_days: int = 7) -> List[Dict[str, Any]]:
        """
        Run all pattern mining analyses.
        
        Args:
            max_age_days: How far back to analyze
            
        Returns:
            Combined list of all discovered patterns
        """
        print(f"🔍 Collecting telemetry data (last {max_age_days} days)...")
        samples = self.collect_data(max_age_days)
        print(f"  Collected {samples} samples")
        
        if samples < self.min_samples:
            print(f"⚠️  Not enough samples for pattern mining (need at least {self.min_samples})")
            return []
        
        all_patterns: List[Dict[str, Any]] = []
        
        # Mine different pattern types
        print(f"⛏️  Mining strategy patterns...")
        all_patterns.extend(self.mine_strategy_patterns())
        
        print(f"⛏️  Mining anchor patterns...")
        all_patterns.extend(self.mine_anchor_patterns())
        
        print(f"⛏️  Mining query length patterns...")
        all_patterns.extend(self.mine_query_length_patterns())
        
        print(f"⛏️  Mining temporal patterns...")
        all_patterns.extend(self.mine_temporal_patterns())
        
        # Store patterns
        self.patterns = all_patterns
        self._save_state()
        
        print(f"✅ Discovered {len(all_patterns)} patterns!")
        
        return all_patterns
    
    def generate_report(self, patterns: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate human-readable report of discovered patterns.
        
        Args:
            patterns: Patterns to report (uses self.patterns if None)
            
        Returns:
            Formatted report string
        """
        patterns = patterns or self.patterns
        
        if not patterns:
            return "No patterns discovered yet. Run mine_all_patterns() first."
        
        report_lines = [
            "=" * 80,
            "ARIA PATTERN MINING REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Patterns: {len(patterns)}",
            "",
            "=" * 80,
            "DISCOVERED PATTERNS",
            "=" * 80,
            ""
        ]
        
        # Group by type
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pattern in patterns:
            by_type[pattern['type']].append(pattern)
        
        for pattern_type, patterns_of_type in by_type.items():
            report_lines.append(f"\n## {pattern_type.replace('_', ' ').title()}")
            report_lines.append("-" * 80)
            
            for i, pattern in enumerate(patterns_of_type, 1):
                report_lines.append(f"\n{i}. {pattern.get('recommendation', 'No recommendation')}")
                
                # Add details
                for key, value in pattern.items():
                    if key not in ['type', 'recommendation']:
                        if isinstance(value, float):
                            report_lines.append(f"   {key}: {value:.3f}")
                        else:
                            report_lines.append(f"   {key}: {value}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return '\n'.join(report_lines)
    
    def export_tuning_recommendations(self, output_file: Path):
        """
        Export patterns as tuning recommendations for auto_tuner.py.
        
        Args:
            output_file: Path to save recommendations
        """
        if not self.patterns:
            print("⚠️  No patterns to export")
            return
        
        recommendations = {
            'generated': datetime.now().isoformat(),
            'patterns': self.patterns,
            'statistics': {
                'total_patterns': len(self.patterns),
                'pattern_types': list(set(p['type'] for p in self.patterns))
            }
        }
        
        output_file.write_text(json.dumps(recommendations, indent=2))
        print(f"💾 Exported {len(self.patterns)} patterns to {output_file}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARIA Pattern Miner - Discover successful retrieval patterns"
    )
    parser.add_argument(
        '--telemetry-dir',
        type=Path,
        default=Path('/media/notapplicable/Internal-SSD/ai-quaternions-model/var/telemetry'),
        help='Telemetry directory'
    )
    parser.add_argument(
        '--max-age-days',
        type=int,
        default=7,
        help='Maximum age of telemetry to analyze (days)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=20,
        help='Minimum samples required for pattern'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Output report file'
    )
    parser.add_argument(
        '--export-recommendations',
        type=Path,
        help='Export tuning recommendations to file'
    )
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = PatternMiner(
        telemetry_dir=args.telemetry_dir,
        min_samples=args.min_samples
    )
    
    # Mine patterns
    patterns = miner.mine_all_patterns(max_age_days=args.max_age_days)
    
    if patterns:
        # Generate report
        report = miner.generate_report(patterns)
        
        if args.report:
            args.report.write_text(report)
            print(f"📄 Report saved to {args.report}")
        else:
            print("\n" + report)
        
        # Export recommendations
        if args.export_recommendations:
            miner.export_tuning_recommendations(args.export_recommendations)
    else:
        print("\nℹ️  No patterns discovered. Try collecting more telemetry data.")


if __name__ == '__main__':
    main()
