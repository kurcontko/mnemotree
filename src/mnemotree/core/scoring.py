from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .models import MemoryType, MemoryItem, EmotionCategory


class DecayFunction(Enum):
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    HYPERBOLIC = "hyperbolic"


class MemoryScoring:
    """Enhanced memory scoring system with advanced algorithms and multiple factors."""
    
    def __init__(
        self,
        importance_weight: float = 0.3,
        recency_weight: float = 0.15,
        access_weight: float = 0.15,
        emotion_weight: float = 0.2,
        context_weight: float = 0.1,
        novelty_weight: float = 0.1,
        #query_weight: float = 0.3,  # New weight for query relevance
        decay_function: DecayFunction = DecayFunction.POWER_LAW,
        score_threshold: float = 0.6,
        learning_rate: float = 0.1
    ):
        self.weights = {
            "importance": importance_weight,
            "recency": recency_weight,
            "access": access_weight,
            "emotional": emotion_weight,
            "context": context_weight,
            "novelty": novelty_weight,
            #"query_relevance": query_weight
        }
        self.decay_function = decay_function
        self.score_threshold = score_threshold
        self.learning_rate = learning_rate
        
    def calculate_memory_score(
        self,
        memory: MemoryItem,
        current_time: datetime,
        semantic_score: Optional[float] = None,
        context_relevance: Optional[float] = None,
        all_memories: Optional[List[MemoryItem]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """Calculate comprehensive memory score using multiple factors."""
        memory_time = datetime.strptime(memory.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
        last_accessed = datetime.strptime(memory.last_accessed, "%Y-%m-%d %H:%M:%S.%f%z")
        
        # Calculate component scores
        scores = {
            "importance": self._calculate_importance_score(memory),
            "recency": self._calculate_recency_score(memory_time, current_time),
            "access": self._calculate_access_score(memory),
            "emotional": self._calculate_emotion_score(memory),
            "context": self._calculate_context_score(memory, context_relevance),
            "connection": self._calculate_connection_score(memory),
            "novelty": self.calculate_novelty_score(memory, all_memories),
            #"query_relevance": self._calculate_query_relevance_score(memory, query_embedding) if query_embedding else 0.5
        }
        
        # Apply adaptive boosting based on memory type
        scores = self._apply_memory_type_boost(scores, memory.memory_type)
        
        # Calculate weighted sum
        total_score = sum(
            self.weights[k] * scores[k] 
            for k in self.weights.keys()
        )
        
        # Incorporate semantic score if provided
        if semantic_score is not None:
            total_score = self._combine_scores(total_score, semantic_score)
            
        return self._normalize_score(total_score)
    
    def _calculate_importance_score(self, memory: MemoryItem) -> float:
        """Calculate importance score with additional factors."""
        base_importance = memory.importance
        
        # Boost importance based on tags
        tag_boost = 0.0
        if memory.tags:
            tag_boost = len(memory.tags) * 0.05
        
        # Consider linked concepts
        concept_boost = 0.0
        if memory.linked_concepts:
            concept_boost = len(memory.linked_concepts) * 0.03
            
        return min(1.0, base_importance + tag_boost + concept_boost)
    
    def _calculate_recency_score(
        self,
        memory_time: datetime,
        current_time: datetime
    ) -> float:
        """
        Calculate recency score using configurable decay functions.
        All functions are normalized to return values between 0 and 1.
        """
        time_diff = (current_time - memory_time).total_seconds()
        base_stability = 24 * 3600  # 24 hours as base stability
        
        if self.decay_function == DecayFunction.EXPONENTIAL:
            # Simple exponential decay
            decay_constant = base_stability
            return math.exp(-time_diff / decay_constant)
        
        elif self.decay_function == DecayFunction.POWER_LAW:
            # Power law decay (Ebbinghaus forgetting curve)
            # Normalized to ensure output is between 0 and 1
            if time_diff < 1:  # Prevent division by zero or negative values
                return 1.0
            
            stability = base_stability
            power = -0.5  # Standard power law decay exponent
            raw_score = (time_diff / stability) ** power
            
            # Clip to ensure maximum is 1.0
            return min(1.0, raw_score)
            
        else:  # HYPERBOLIC
            # Hyperbolic decay
            decay_factor = base_stability
            return 1 / (1 + time_diff / decay_factor)
    
    def _calculate_access_score(self, memory: MemoryItem) -> float:
        """Calculate access score with temporal patterns."""
        recent_access_weight = 0.7
        pattern_weight = 0.3
        
        # Basic access frequency score
        base_score = self._calculate_base_access_score(memory.access_count)
        
        # Consider recent access patterns
        recent_score = self._analyze_recent_access_patterns(memory)
        
        return (recent_access_weight * base_score + 
                pattern_weight * recent_score)
    
    def _calculate_context_score(
        self,
        memory: MemoryItem,
        context_relevance: Optional[float]
    ) -> float:
        """Calculate contextual relevance score."""
        if context_relevance is not None:
            return context_relevance
            
        # Calculate based on context richness\
        if not memory.tags:
            return 0.5
        context_factors = [
            #len(memory.context) * 0.1,  # Context completeness
            len(memory.tags) * 0.1,    # Tag relevance
            0.5  # Base context score
        ]
        
        return min(1.0, sum(context_factors))
    
    def _calculate_connection_score(self, memory: MemoryItem) -> float:
        """Calculate score based on memory connections."""
        associations = len(memory.associations) if memory.associations else 0
        conflicts = len(memory.conflicts_with) if memory.conflicts_with else 0
        connection_count = associations + conflicts
        
        # Log scale for connection score
        if connection_count == 0:
            return 0.5
        return min(1.0, 0.5 + math.log(connection_count + 1) * 0.2)
    
    def _apply_memory_type_boost(
        self,
        scores: Dict[str, float],
        memory_type: MemoryType
    ) -> Dict[str, float]:
        """Apply memory type-specific score boosting."""
        boost_factors = {
            MemoryType.EPISODIC: {"emotional": 1.2, "recency": 1.1},
            MemoryType.SEMANTIC: {"importance": 1.2, "connection": 1.1},
            MemoryType.PROCEDURAL: {"access": 1.2, "context": 1.1},
            MemoryType.WORKING: {"recency": 1.3, "importance": 1.1},
            MemoryType.AUTOBIOGRAPHICAL: {"emotional": 1.2, "context": 1.1},
            MemoryType.PROSPECTIVE: {"importance": 1.2, "context": 1.1},
            MemoryType.CONDITIONING: {"access": 1.2, "connection": 1.1},
            MemoryType.PRIMING: {"access": 1.2, "connection": 1.1}
        }
        
        if memory_type in boost_factors:
            for component, factor in boost_factors[memory_type].items():
                scores[component] *= factor
                
        return scores
    
    def _combine_scores(self, base_score: float, semantic_score: float) -> float:
        """Combine base and semantic scores using geometric mean."""
        return math.sqrt(base_score * semantic_score)
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range using sigmoid function."""
        return 1 / (1 + math.exp(-5 * (score - 0.5)))
    
    def update_weights(self, feedback_data: List[Tuple[MemoryItem, float]]) -> None:
        """Update scoring weights based on feedback using gradient descent."""
        for memory, target_score in feedback_data:
            predicted_score = self.calculate_memory_score(
                memory,
                datetime.now(timezone.utc)
            )
            error = target_score - predicted_score
            
            # Update weights using gradient descent
            for component, weight in self.weights.items():
                gradient = error * getattr(
                    self,
                    f"_calculate_{component}_score"
                )(memory)
                self.weights[component] += self.learning_rate * gradient
                
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
    
    def _calculate_base_access_score(self, access_count: int) -> float:
        """
        Calculate base access frequency score using a logarithmic scale with
        diminishing returns and normalized to [0, 1].
        
        Args:
            access_count: Number of times the memory has been accessed
            
        Returns:
            Float between 0 and 1 representing the access frequency score
        """
        if access_count == 0:
            return 0.0
            
        # Parameters for tuning the logarithmic curve
        base = 2
        scaling_factor = 0.3
        max_count = 100  # Theoretical maximum for normalization
        
        # Calculate score using log scale with diminishing returns
        raw_score = math.log(access_count + 1, base) * scaling_factor
        max_score = math.log(max_count + 1, base) * scaling_factor
        
        # Normalize to [0, 1]
        return min(1.0, raw_score / max_score)

    def _analyze_recent_access_patterns(self, memory: MemoryItem) -> float:
        """
        Analyze temporal patterns in memory access with spaced repetition principles.
        """
        if not memory.access_history:
            return 0.0
            
        current_time = datetime.now(timezone.utc)
        access_times = [
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
            for timestamp in memory.access_history
        ]
        
        # Calculate time windows and their weights
        time_windows = {
            'last_hour': (timedelta(hours=1), 0.35),
            'last_day': (timedelta(days=1), 0.25),
            'last_week': (timedelta(weeks=1), 0.2),
            'last_month': (timedelta(days=30), 0.2)
        }
        
        window_scores = {}
        for window_name, (duration, _) in time_windows.items():
            cutoff_time = current_time - duration
            recent_accesses = sum(1 for t in access_times if t >= cutoff_time)
            
            # Calculate spaced repetition optimal interval
            if recent_accesses > 0:
                interval = duration.total_seconds() / (recent_accesses + 1)
                optimal_interval = self._calculate_optimal_interval(recent_accesses)
                interval_score = math.exp(-abs(interval - optimal_interval) / optimal_interval)
            else:
                interval_score = 0.0
                
            window_scores[window_name] = interval_score
        
        # Calculate regularity bonus
        regularity_score = self._calculate_access_regularity(access_times)
        
        # Combine scores with weights
        final_score = sum(
            score * weight 
            for (_, weight), score in zip(time_windows.values(), window_scores.values())
        )
        final_score += regularity_score * 0.2  # Add regularity bonus
        
        return min(1.0, final_score)

    def _calculate_optimal_interval(self, review_count: int) -> float:
        """Calculate optimal interval between reviews based on spaced repetition."""
        base_interval = 24 * 3600  # 24 hours in seconds
        return base_interval * (1.5 ** (review_count - 1))

    def _calculate_access_regularity(self, access_times: List[datetime]) -> float:
        """Calculate how regular/consistent the access pattern is."""
        if len(access_times) < 2:
            return 0.0
            
        # Sort access times and calculate intervals
        sorted_times = sorted(access_times)
        intervals = [
            (t2 - t1).total_seconds() / 3600  # Convert to hours
            for t1, t2 in zip(sorted_times[:-1], sorted_times[1:])
        ]
        
        if not intervals:
            return 0.0
            
        # Calculate coefficient of variation (CV) of intervals
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.0
            
        cv = std_interval / mean_interval
        
        # Convert CV to a score between 0 and 1
        # Lower CV (more regular) gives higher score
        regularity_score = math.exp(-cv)
        
        return regularity_score
    
    def calculate_novelty_score(self, memory: MemoryItem, all_memories: List[MemoryItem]) -> float:
        """Calculate novelty score based on similarity to other memories."""
        if not all_memories:
            return 0.5  # Default score if no other memories exist
            
        similarities = []
        for other in all_memories:
            if other.memory_id != memory.memory_id:
                sim = self._cosine_similarity(memory.embedding, other.embedding)
                similarities.append(sim)
                
        if not similarities:
            return 0.5
            
        avg_similarity = sum(similarities) / len(similarities)
        novelty_score = 1.0 - avg_similarity
        
        # Apply sigmoid to smooth the novelty score
        return 1 / (1 + math.exp(-5 * (novelty_score - 0.5)))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_emotion_score(self, memory: MemoryItem) -> float:
        """
        Calculate emotion score based on EmotionalContext combining direct valence/arousal
        with emotional category valence mapping.
        
        Uses:
        - Direct valence/arousal from EmotionalContext
        - Mapped valence scores for emotion categories as supplementary data
        - Number and types of emotions present
        """
        if not memory.emotions:
            return 0.5  # Neutral score for memories without emotional context
        
        # Emotion category valence mapping
        valence_scores = {
            EmotionCategory.JOY: 1.0,
            EmotionCategory.SADNESS: -0.7,
            EmotionCategory.ANGER: -0.8,
            EmotionCategory.FEAR: -0.6,
            EmotionCategory.SURPRISE: 0.3,
            EmotionCategory.DISGUST: -0.5,
            EmotionCategory.TRUST: 0.8,
            EmotionCategory.ANTICIPATION: 0.4,
            EmotionCategory.NEUTRAL: 0.0
        }
        
        # Get explicit valence/arousal from EmotionalContext
        explicit_valence = memory.emotional_valence if memory.emotional_valence is not None else None
        arousal = memory.emotional_arousal if memory.emotional_arousal is not None else 0.5
        
        # Calculate valence score - use explicit if available, otherwise use mapped values
        if explicit_valence is not None:
            valence_score = (explicit_valence + 1) / 2  # Convert from [-1, 1] to [0, 1]
        elif memory.emotions:
            # Calculate average valence from emotion categories
            emotion_valences = [
                valence_scores.get(emotion, 0.0) if isinstance(emotion, EmotionCategory)
                else valence_scores.get(EmotionCategory(emotion), 0.0)
                for emotion in memory.emotions
            ]
            avg_valence = sum(emotion_valences) / len(emotion_valences)
            valence_score = (avg_valence + 1) / 2  # Convert to [0, 1]
        else:
            valence_score = 0.5  # Neutral if no valence information available
        
        # Calculate emotion presence score
        emotion_presence_score = 0.5
        if memory.emotions:
            num_emotions = len(memory.emotions)
            emotion_presence_score = min(1.0, 0.5 + (num_emotions * 0.1))
        
        # Combine scores with weights
        weights = {
            'valence': 0.4,
            'arousal': 0.4,
            'emotion_presence': 0.2
        }
        
        final_score = (
            weights['valence'] * valence_score +
            weights['arousal'] * arousal +
            weights['emotion_presence'] * emotion_presence_score
        )
        
        # Apply sigmoid function for smoother distribution
        return 1 / (1 + math.exp(-5 * (final_score - 0.5)))
    
    def filter_memories_by_score(
        self,
        memories: List[MemoryItem],
        semantic_scores: Optional[List[float]] = None,
        context_relevance: Optional[List[float]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[MemoryItem]:
        """Filter memories based on score threshold."""
        current_time = datetime.now(timezone.utc)
        filtered_memories = []
        for i, memory in enumerate(memories):
            semantic_score = semantic_scores[i] if semantic_scores else None
            score = self.calculate_memory_score(
                memory, 
                current_time, 
                semantic_score,
                context_relevance,
                memories,
                query_embedding
            )
            if score >= self.score_threshold:
                filtered_memories.append(memory)
        return filtered_memories
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
    
    def _calculate_query_relevance_score(self, memory: MemoryItem, query_embedding: List[float]) -> float:
        """Calculate how relevant the memory is to the current query using embeddings."""
        if not query_embedding or not memory.embedding:
            return 0.5
        similarity = self._cosine_similarity(memory.embedding, query_embedding)
        # Convert similarity [0,1] to a smoother scale if desired
        return 1 / (1 + math.exp(-5 * (similarity - 0.5)))
