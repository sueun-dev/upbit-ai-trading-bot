"""Global analysis state manager for real-time dashboard updates."""

from datetime import datetime
from typing import Dict, Any


class AnalysisStateManager:
    """Manager for tracking analysis pipeline state."""
    
    def __init__(self):
        self._state = {
            'stage': 'idle',
            'progress': 0,
            'status': 'Waiting for next cycle',
            'last_update': datetime.now().isoformat(),
            'stages': {
                'news_collection': {'progress': 0, 'status': 'Waiting'},
                'symbol_extraction': {'progress': 0, 'status': 'Waiting'},
                'market_analysis': {'progress': 0, 'status': 'Waiting'},
                'ai_decision': {'progress': 0, 'status': 'Waiting'}
            },
            'decisions_count': 0,
            'cycle_duration': 0
        }
    
    # USED
    def update_stage(self, stage: str, progress: int, status: str):
        """Update current analysis stage."""
        self._state['stage'] = stage
        self._state['progress'] = progress
        self._state['status'] = status
        self._state['last_update'] = datetime.now().isoformat()
        
        # Update specific stage progress
        if stage in self._state['stages']:
            self._state['stages'][stage]['progress'] = progress
            self._state['stages'][stage]['status'] = status
    
    def complete_cycle(self, decisions_count: int, duration: float):
        """Mark cycle as completed."""
        self._state['stage'] = 'completed'
        self._state['progress'] = 100
        self._state['status'] = f'Cycle completed - {decisions_count} decisions in {duration:.1f}s'
        self._state['last_update'] = datetime.now().isoformat()
        self._state['decisions_count'] = decisions_count
        self._state['cycle_duration'] = duration
        
        # Mark all stages as completed
        for stage in self._state['stages']:
            self._state['stages'][stage]['progress'] = 100
            self._state['stages'][stage]['status'] = 'Completed'
    
    # USED
    def reset_cycle(self):
        """Reset for new cycle."""
        self._state['stage'] = 'starting'
        self._state['progress'] = 0
        self._state['status'] = 'Starting new cycle...'
        self._state['last_update'] = datetime.now().isoformat()
        
        # Reset all stages
        for stage in self._state['stages']:
            self._state['stages'][stage]['progress'] = 0
            self._state['stages'][stage]['status'] = 'Waiting'


# Global instance
analysis_state = AnalysisStateManager()