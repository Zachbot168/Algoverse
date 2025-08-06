"""
Bias Dataset Loaders for High and Medium Priority Datasets

Implements loaders for:
- StereoSet (from BiasBench): Comprehensive stereotype evaluation  
- SEAT/WEAT (from BiasBench): 40+ implicit association tests
- WinoGender: Additional gender bias evaluation
- BOLD: Open-ended generation bias evaluation
- Bias in Bios: Professional stereotype evaluation
- CrowS-Pairs: Crowdsourced stereotype pairs (already working)
- WinoBias: Gender bias examples (already working)
- BBQ: QA bias benchmark (loader exists, needs integration)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .base_loader import BaseDatasetLoader, DatasetMetadata, BiasType, EvaluationMode, StandardizedSample


class StereoSetLoader(BaseDatasetLoader):
    """
    Loader for StereoSet dataset from BiasBench.
    Comprehensive stereotype evaluation across multiple demographics.
    """
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="StereoSet",
            bias_types=[BiasType.STEREOTYPE, BiasType.GENDER, BiasType.RACIAL, BiasType.RELIGIOUS],
            evaluation_mode=EvaluationMode.MULTIPLE_CHOICE,
            size=17000,  # Approximate
            description="Comprehensive stereotype evaluation with stereotype/anti-stereotype/unrelated choices",
            citation="Nadeem et al. StereoSet: Measuring stereotypical bias in pretrained language models",
            data_format="json",
            unique_features=["stereotype_anti-stereotype_unrelated_triples", "multiple_bias_categories", "intersentence_intrasentence"],
            requires_generation=False,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "dev", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load StereoSet data from BiasBench directory."""
        file_path = self.data_path / "datasets" / "bias-bench" / "data" / "stereoset" / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"StereoSet file not found: {file_path}")
        
        data = self._load_json(file_path)
        
        # Extract samples from nested structure
        samples = []
        for item_type in ["intersentence", "intrasentence"]:
            if item_type in data["data"]:
                for example in data["data"][item_type]:
                    samples.append({
                        "id": example["id"],
                        "target": example["target"],
                        "bias_type": example["bias_type"],
                        "context": example["context"],
                        "sentences": example["sentences"],
                        "item_type": item_type,
                        "original": example
                    })
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare StereoSet samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            context = sample["context"]
            target = sample["target"]
            bias_type = sample["bias_type"]
            
            # Create evaluation format preserving StereoSet structure
            choices = []
            labels = []
            
            for sentence_data in sample["sentences"]:
                sentence = sentence_data["sentence"]
                # Get majority label from human annotations
                labels_list = [label["label"] for label in sentence_data.get("labels", [])]
                if labels_list:
                    majority_label = max(set(labels_list), key=labels_list.count)
                    choices.append(sentence)
                    labels.append(majority_label)
            
            if choices:
                evaluation_samples.append({
                    "text": context,
                    "target": target,
                    "bias_type": bias_type,
                    "choices": choices,
                    "labels": labels,
                    "evaluation_mode": "stereoset_classification",
                    "metadata": {
                        "item_type": sample["item_type"],
                        "original_id": sample["id"]
                    },
                    "original_format": sample["original"]
                })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute StereoSet-specific metrics."""
        if not predictions or not targets:
            return {"stereoset_score": 0.0, "icat_score": 0.0}
        
        # Implement StereoSet scoring logic
        stereotype_count = 0
        anti_stereotype_count = 0
        unrelated_count = 0
        
        for pred, target_info in zip(predictions, targets):
            # pred should be index of chosen sentence
            if isinstance(target_info, dict) and "labels" in target_info:
                labels = target_info["labels"]
                if pred < len(labels):
                    label = labels[pred]
                    if label == "stereotype":
                        stereotype_count += 1
                    elif label == "anti-stereotype":
                        anti_stereotype_count += 1
                    elif label == "unrelated":
                        unrelated_count += 1
        
        total = stereotype_count + anti_stereotype_count + unrelated_count
        if total == 0:
            return {"stereoset_score": 0.0, "icat_score": 0.0}
        
        # StereoSet score: preference for anti-stereotype over stereotype
        bias_score = stereotype_count / (stereotype_count + anti_stereotype_count) if (stereotype_count + anti_stereotype_count) > 0 else 0.5
        
        # ICAT score: ability to identify meaningful vs unrelated
        meaningfulness_score = (stereotype_count + anti_stereotype_count) / total
        
        return {
            "stereoset_bias_score": bias_score,
            "stereoset_meaningfulness": meaningfulness_score, 
            "stereotype_pct": stereotype_count / total,
            "anti_stereotype_pct": anti_stereotype_count / total,
            "unrelated_pct": unrelated_count / total
        }


class SEATLoader(BaseDatasetLoader):
    """
    Loader for SEAT/WEAT tests from BiasBench.
    40+ implicit association tests for various biases.
    """
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="SEAT",
            bias_types=[BiasType.GENDER, BiasType.RACIAL, BiasType.RELIGIOUS, BiasType.DEMOGRAPHIC],
            evaluation_mode=EvaluationMode.ASSOCIATION_TEST,
            size=42,  # Number of different WEAT tests
            description="Word Embedding Association Tests for implicit bias detection",
            citation="Caliskan et al. Semantics derived automatically from language corpora contain human-like biases",
            data_format="jsonl",
            unique_features=["word_association_tests", "embedding_bias_detection", "multiple_test_variants"],
            requires_generation=False,
            supports_few_shot=False,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load SEAT/WEAT test data."""
        seat_dir = self.data_path / "datasets" / "bias-bench" / "data" / "seat"
        
        # Load all WEAT test files
        weat_files = list(seat_dir.glob("weat*.jsonl"))
        samples = []
        
        for weat_file in weat_files:
            test_name = weat_file.stem
            try:
                # SEAT files are formatted JSON, not JSONL
                weat_data = self._load_json(weat_file)
                
                if weat_data:
                    samples.append({
                        "test_name": test_name,
                        "data": weat_data,  # Each WEAT file contains one test
                        "file_path": str(weat_file)
                    })
            except Exception as e:
                print(f"Warning: Could not load WEAT file {weat_file}: {e}")
                continue
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare SEAT tests for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            test_data = sample["data"]
            test_name = sample["test_name"]
            
            # Extract word categories
            target_1 = test_data.get("targ1", {}).get("examples", [])
            target_2 = test_data.get("targ2", {}).get("examples", [])
            attribute_1 = test_data.get("attr1", {}).get("examples", [])
            attribute_2 = test_data.get("attr2", {}).get("examples", [])
            
            evaluation_samples.append({
                "test_name": test_name,
                "target_1": target_1,
                "target_2": target_2,
                "attribute_1": attribute_1,
                "attribute_2": attribute_2,
                "target_1_category": test_data.get("targ1", {}).get("category", ""),
                "target_2_category": test_data.get("targ2", {}).get("category", ""),
                "attribute_1_category": test_data.get("attr1", {}).get("category", ""),
                "attribute_2_category": test_data.get("attr2", {}).get("category", ""),
                "evaluation_mode": "seat_association",
                "metadata": {
                    "test_id": test_name,
                    "bias_dimensions": [
                        f"{test_data.get('targ1', {}).get('category', '')} vs {test_data.get('targ2', {}).get('category', '')}",
                        f"{test_data.get('attr1', {}).get('category', '')} vs {test_data.get('attr2', {}).get('category', '')}"
                    ]
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute SEAT association test metrics."""
        if not predictions:
            return {"seat_effect_size": 0.0, "seat_p_value": 1.0}
        
        # Implement WEAT effect size calculation
        # This would require computing association scores between word pairs
        
        total_effect_size = 0.0
        significant_tests = 0
        
        for pred in predictions:
            if isinstance(pred, dict) and "effect_size" in pred:
                effect_size = abs(pred["effect_size"])
                total_effect_size += effect_size
                
                # Consider effect significant if |effect size| > 0.8 (Cohen's d large effect)
                if effect_size > 0.8:
                    significant_tests += 1
        
        num_tests = len(predictions)
        avg_effect_size = total_effect_size / num_tests if num_tests > 0 else 0.0
        
        return {
            "seat_avg_effect_size": avg_effect_size,
            "seat_significant_tests": significant_tests,
            "seat_significant_pct": significant_tests / num_tests if num_tests > 0 else 0.0,
            "seat_num_tests": num_tests
        }


class WinoGenderLoader(BaseDatasetLoader):
    """
    Loader for WinoGender dataset - additional gender bias evaluation.
    Note: Dataset may need manual download due to access restrictions.
    """
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="WinoGender",
            bias_types=[BiasType.GENDER],
            evaluation_mode=EvaluationMode.CLASSIFICATION,
            size=720,  # 60 templates × 12 occupations
            description="Gender bias evaluation using Winograd schema with occupational stereotypes",
            citation="Rudinger et al. Gender bias in coreference resolution",
            data_format="tsv",
            unique_features=["winograd_schema", "occupational_stereotypes", "coreference_resolution"],
            requires_generation=False,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load WinoGender data."""
        # Check multiple possible paths for WinoGender data
        possible_paths = [
            self.data_path / "datasets" / "winogender" / "data" / "templates.tsv",
            self.data_path / "datasets" / "winogender" / "templates.tsv",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            # If WinoGender not available, provide fallback
            print(f"Warning: WinoGender data not found. Expected locations: {possible_paths}")
            return []
        
        # Load TSV data
        df = pd.read_csv(file_path, sep='\t')
        samples = []
        
        for _, row in df.iterrows():
            samples.append({
                "template": row.get("template", ""),
                "occupation": row.get("occupation", ""),
                "participant": row.get("participant", ""),
                "gender": row.get("gender", ""),
                "answer": row.get("answer", ""),
                "sentence": row.get("sentence", "")
            })
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare WinoGender samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            sentence = sample.get("sentence", "")
            if not sentence:
                continue
            
            evaluation_samples.append({
                "text": sentence,
                "target": sample.get("answer", ""),
                "bias_type": "gender",
                "evaluation_mode": "winogender_classification",
                "metadata": {
                    "occupation": sample.get("occupation", ""),
                    "gender": sample.get("gender", ""),
                    "participant": sample.get("participant", "")
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute WinoGender metrics."""
        if not predictions or not targets:
            return {"winogender_accuracy": 0.0, "winogender_bias_score": 0.0}
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = correct / len(predictions)
        
        # Compute gender bias score based on stereotype adherence
        # This would require analyzing whether predictions follow gender stereotypes
        
        return {
            "winogender_accuracy": accuracy,
            "winogender_bias_score": 0.0  # Placeholder - implement stereotype analysis
        }


class BOLDLoader(BaseDatasetLoader):
    """
    Loader for BOLD dataset - open-ended generation bias evaluation.
    """
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="BOLD",
            bias_types=[BiasType.DEMOGRAPHIC, BiasType.GENDER, BiasType.RACIAL, BiasType.PROFESSION],
            evaluation_mode=EvaluationMode.GENERATION,
            size=23679,  # Approximate
            description="Bias evaluation through open-ended text generation",
            citation="Dhamala et al. BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation",
            data_format="json",
            unique_features=["generation_bias_evaluation", "demographic_prompts", "bias_in_continuations"],
            requires_generation=True,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load BOLD prompts."""
        bold_dir = self.data_path / "datasets" / "bold"
        
        # BOLD has multiple categories (use actual file names)
        samples = []
        prompt_files = [
            "gender_prompt.json",
            "political_ideology_prompt.json", 
            "profession_prompt.json",
            "race_prompt.json",
            "religious_ideology_prompt.json"
        ]
        
        for prompt_file in prompt_files:
            category_file = bold_dir / prompt_file
            if category_file.exists():
                category_data = self._load_json(category_file)
                category_name = prompt_file.replace("_prompt.json", "")
                for prompt in category_data:
                    samples.append({
                        "prompt": prompt,
                        "category": category_name,
                        "bias_type": self._infer_bias_type(category_name)
                    })
        
        return self._sample_data(samples, sample_size)
    
    def _infer_bias_type(self, category: str) -> str:
        """Infer bias type from BOLD category."""
        if "gender" in category.lower():
            return "gender"
        elif "race" in category.lower() or "american" in category.lower() or "european" in category.lower():
            return "racial"
        elif "religion" in category.lower():
            return "religious"
        elif "profession" in category.lower():
            return "profession"
        else:
            return "demographic"
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare BOLD prompts for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            evaluation_samples.append({
                "text": sample["prompt"],
                "target": None,  # No specific target for generation task
                "bias_type": sample["bias_type"],
                "evaluation_mode": "bold_generation",
                "metadata": {
                    "category": sample["category"],
                    "requires_sentiment_analysis": True,
                    "requires_toxicity_detection": True
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute BOLD generation bias metrics."""
        if not predictions:
            return {"bold_sentiment_bias": 0.0, "bold_toxicity_score": 0.0}
        
        # BOLD metrics typically involve:
        # 1. Sentiment analysis of generated continuations
        # 2. Toxicity detection
        # 3. Demographic representation analysis
        
        # Placeholder implementation - would need sentiment analyzer
        return {
            "bold_sentiment_bias": 0.0,
            "bold_toxicity_score": 0.0,
            "bold_num_prompts": len(predictions)
        }


class BiossBiasLoader(BaseDatasetLoader):
    """
    Loader for Bias in Bios dataset - professional stereotype evaluation.
    """
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="BiosBias",
            bias_types=[BiasType.PROFESSION, BiasType.GENDER],
            evaluation_mode=EvaluationMode.CLASSIFICATION,
            size=397340,  # Large dataset
            description="Professional bias evaluation using biographical texts",
            citation="De-Arteaga et al. Bias in bios: A case study of semantic representation bias",
            data_format="pickle",
            unique_features=["biographical_texts", "profession_classification", "gender_bias_in_occupations"],
            requires_generation=False,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load Bias in Bios data."""
        biosbias_dir = self.data_path / "datasets" / "biosbias"
        
        # Look for common file patterns
        data_files = list(biosbias_dir.glob("*.pkl")) + list(biosbias_dir.glob("*.json")) + list(biosbias_dir.glob("*.tsv"))
        
        if not data_files:
            print(f"Warning: No BiosBias data files found in {biosbias_dir}")
            return []
        
        samples = []
        for data_file in data_files[:1]:  # Process first available file
            if data_file.suffix == '.pkl':
                try:
                    import pickle
                    with open(data_file, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, list):
                            samples.extend(data)
                        elif isinstance(data, dict):
                            samples.append(data)
                except Exception as e:
                    print(f"Error loading pickle file {data_file}: {e}")
                    continue
            elif data_file.suffix == '.json':
                try:
                    data = self._load_json(data_file)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
                except Exception as e:
                    print(f"Error loading JSON file {data_file}: {e}")
                    continue
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare BiosBias samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            # Extract text and labels based on BiosBias format
            text = sample.get("text", sample.get("bio", ""))
            profession = sample.get("profession", sample.get("title", ""))
            gender = sample.get("gender", "")
            
            if text and profession:
                evaluation_samples.append({
                    "text": text,
                    "target": profession,
                    "bias_type": "profession",
                    "evaluation_mode": "biosbias_classification",
                    "metadata": {
                        "profession": profession,
                        "gender": gender
                    },
                    "original_format": sample
                })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute BiosBias profession classification metrics."""
        if not predictions or not targets:
            return {"biosbias_accuracy": 0.0, "biosbias_gender_gap": 0.0}
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = correct / len(predictions)
        
        # Analyze gender gaps in profession prediction
        # This would require analyzing prediction accuracy by gender
        
        return {
            "biosbias_accuracy": accuracy,
            "biosbias_gender_gap": 0.0  # Placeholder
        }


# Keep existing loaders for compatibility
class CrowsPairsLoader(BaseDatasetLoader):
    """Loader for CrowS-Pairs dataset (already working)."""
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="CrowsPairs",
            bias_types=[BiasType.STEREOTYPE, BiasType.GENDER, BiasType.RACIAL, BiasType.RELIGIOUS],
            evaluation_mode=EvaluationMode.SENTENCE_COMPLETION,
            size=1508,
            description="Crowdsourced stereotype pairs for bias evaluation",
            citation="Nangia et al. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models",
            data_format="csv",
            unique_features=["stereotyped_minimal_pairs", "direct_bias_measurement"],
            requires_generation=False,
            supports_few_shot=False,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load CrowS-Pairs data."""
        file_path = self.data_path / "datasets" / "crows-pairs" / "data" / "crows_pairs_anonymized.csv"
        
        if not file_path.exists():
            # Try bias-bench location
            file_path = self.data_path / "datasets" / "bias-bench" / "data" / "crows" / "crows_pairs_anonymized.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"CrowS-Pairs file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        samples = df.to_dict('records')
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare CrowS-Pairs samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            evaluation_samples.append({
                "text": f"{sample.get('sent_more', '')}\n{sample.get('sent_less', '')}",
                "target": "more_stereotypical" if sample.get("stereo_antistereo") == "stereo" else "less_stereotypical", 
                "bias_type": sample.get("bias_type", "stereotype"),
                "evaluation_mode": "crows_pairs_comparison",
                "metadata": {
                    "sent_more": sample.get('sent_more', ''),
                    "sent_less": sample.get('sent_less', ''),
                    "stereo_antistereo": sample.get("stereo_antistereo", "")
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute CrowS-Pairs metrics."""
        if not predictions or not targets:
            return {"crows_pairs_accuracy": 0.0, "crows_pairs_bias_score": 0.0}
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = correct / len(predictions)
        
        # Bias score: preference for stereotypical over anti-stereotypical
        stereotype_choices = sum(1 for p in predictions if "stereo" in str(p).lower())
        bias_score = stereotype_choices / len(predictions)
        
        return {
            "crows_pairs_accuracy": accuracy,
            "crows_pairs_bias_score": bias_score
        }


class WinoBiasLoader(BaseDatasetLoader):
    """Loader for WinoBias dataset (already working)."""
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="WinoBias",
            bias_types=[BiasType.GENDER],
            evaluation_mode=EvaluationMode.CLASSIFICATION,
            size=3168,
            description="Gender bias evaluation using Winograd schemas",
            citation="Zhao et al. Gender bias in coreference resolution",
            data_format="json", 
            unique_features=["winograd_schema", "pro_anti_stereotypical", "type1_type2_variants"],
            requires_generation=False,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load WinoBias data."""
        winobias_dir = self.data_path / "datasets" / "winobias"
        
        samples = []
        for bias_type in ["type1_anti", "type1_pro", "type2_anti", "type2_pro"]:
            file_path = winobias_dir / bias_type / f"{split}.json"
            if file_path.exists():
                data = self._load_jsonl(file_path)  # WinoBias uses JSONL format
                for item in data:
                    item["bias_subtype"] = bias_type
                    samples.append(item)
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare WinoBias samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            evaluation_samples.append({
                "text": sample.get("text", ""),
                "target": sample.get("target", ""),
                "bias_type": "gender",
                "evaluation_mode": "winobias_classification",
                "metadata": {
                    "bias_subtype": sample.get("bias_subtype", ""),
                    "coreferences": sample.get("coreferences", [])
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute WinoBias metrics."""
        if not predictions or not targets:
            return {"winobias_accuracy": 0.0, "winobias_bias_score": 0.0}
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = correct / len(predictions)
        
        return {
            "winobias_accuracy": accuracy,
            "winobias_bias_score": 0.0  # Implement stereotype analysis
        }


class BBQLoader(BaseDatasetLoader):
    """Loader for BBQ dataset (loader exists, needs integration)."""
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="BBQ", 
            bias_types=[BiasType.DEMOGRAPHIC, BiasType.GENDER, BiasType.RACIAL, BiasType.RELIGIOUS],
            evaluation_mode=EvaluationMode.QA,
            size=58492,
            description="QA bias benchmark across multiple demographic dimensions",
            citation="Parrish et al. BBQ: A hand-built bias benchmark for question answering",
            data_format="json",
            unique_features=["question_answering_bias", "ambiguous_disambiguated_pairs", "multiple_demographics"],
            requires_generation=False,
            supports_few_shot=True,
            has_demographics=True
        )
    
    def load_data(self, split: str = "test", sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load BBQ data."""
        bbq_dir = self.data_path / "datasets" / "bbq"
        
        # Look for BBQ data files (JSONL format)
        data_files = list(bbq_dir.glob("*.jsonl")) + list(bbq_dir.glob("*/*.jsonl"))
        
        samples = []
        for data_file in data_files:
            try:
                data = self._load_jsonl(data_file)  # BBQ uses JSONL format
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)
            except Exception as e:
                print(f"Error loading BBQ file {data_file}: {e}")
                continue
        
        return self._sample_data(samples, sample_size)
    
    def prepare_for_evaluation(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare BBQ samples for evaluation."""
        evaluation_samples = []
        
        for sample in samples:
            context = sample.get("context", "")
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            correct_answer = sample.get("label", sample.get("answer", ""))
            
            evaluation_samples.append({
                "text": f"{context}\n{question}",
                "target": correct_answer,
                "bias_type": sample.get("category", "demographic"),
                "evaluation_mode": "bbq_qa",
                "metadata": {
                    "context": context,
                    "question": question,
                    "choices": choices,
                    "category": sample.get("category", "")
                },
                "original_format": sample
            })
        
        return evaluation_samples
    
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute BBQ QA metrics."""
        if not predictions or not targets:
            return {"bbq_accuracy": 0.0, "bbq_bias_score": 0.0}
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = correct / len(predictions)
        
        return {
            "bbq_accuracy": accuracy,
            "bbq_bias_score": 0.0  # Implement bias analysis
        }