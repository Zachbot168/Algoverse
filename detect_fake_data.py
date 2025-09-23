#!/usr/bin/env python3
"""
Fake Data Detection Script for Algoverse

This script scans the codebase for patterns that indicate fake data generation,
mock results, or simulated outputs that should not be present in research code.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

class FakeDataDetector:
    """Detects fake data patterns in the codebase."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.fake_patterns = {
            # Random number generation in results
            "random_generation": [
                r"np\.random\.random\(\)",
                r"np\.random\.uniform\(",
                r"np\.random\.randint\(",
                r"random\.random\(\)",
                r"random\.uniform\(",
                r"random\.randint\(",
            ],
            
            # Hardcoded result values
            "hardcoded_values": [
                r"accuracy.*=.*0\.[567]\d*",  # Suspicious accuracy values
                r"bias_score.*=.*0\.[34]\d*",  # Suspicious bias scores
                r"importance.*=.*0\.[89]\d*",  # Suspicious importance scores
            ],
            
            # Suspicious fallback patterns
            "suspicious_fallbacks": [
                r"return.*np\.zeros\(\d+\)",
                r"return.*np\.ones\(\d+\)",
                r"return.*\{[^}]*accuracy.*0\.5[^}]*\}",
            ],
            
            # Mock/fake indicators in comments or variables
            "mock_indicators": [
                r"#.*mock",
                r"#.*fake",
                r"#.*simulate",
                r"#.*placeholder",
                r"mock_\w+",
                r"fake_\w+",
                r"simulate_\w+",
            ]
        }
        
        # Files to exclude from checking
        self.exclude_patterns = [
            r".*test.*\.py$",  # Test files may legitimately use mock data
            r".*demo.*\.py$",  # Demo files may use placeholder data
            r".*example.*\.py$",  # Example files may use fake data
            r"detect_fake_data\.py$",  # This script itself
        ]
        
        self.issues_found = []
    
    def should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded from analysis."""
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return True
        return False
    
    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for fake data patterns."""
        if self.should_exclude_file(str(file_path)):
            return []
        
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                
                for category, patterns in self.fake_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append({
                                'file': str(file_path),
                                'line': line_num,
                                'category': category,
                                'pattern': pattern,
                                'content': line_stripped,
                                'severity': self._get_severity(category)
                            })
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for a category."""
        severity_map = {
            "random_generation": "CRITICAL",
            "hardcoded_values": "HIGH", 
            "suspicious_fallbacks": "MEDIUM",
            "mock_indicators": "LOW"
        }
        return severity_map.get(category, "UNKNOWN")
    
    def scan_directory(self, directory: Path) -> None:
        """Scan all Python files in a directory."""
        python_files = list(directory.rglob("*.py"))
        
        print(f"🔍 Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            file_issues = self.scan_file(file_path)
            self.issues_found.extend(file_issues)
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of findings."""
        if not self.issues_found:
            return "✅ No fake data patterns detected!"
        
        # Group by severity
        by_severity = {}
        for issue in self.issues_found:
            severity = issue['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)
        
        report = "🚨 FAKE DATA DETECTION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        total_issues = len(self.issues_found)
        report += f"Total issues found: {total_issues}\n\n"
        
        # Report by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                issues = by_severity[severity]
                report += f"{severity} ISSUES ({len(issues)}):\n"
                report += "-" * 30 + "\n"
                
                for issue in issues:
                    report += f"File: {issue['file']}\n"
                    report += f"Line: {issue['line']}\n"
                    report += f"Category: {issue['category']}\n"
                    report += f"Pattern: {issue['pattern']}\n"
                    report += f"Code: {issue['content']}\n"
                    report += "\n"
        
        # Summary by file
        file_counts = {}
        for issue in self.issues_found:
            file_path = issue['file']
            if file_path not in file_counts:
                file_counts[file_path] = 0
            file_counts[file_path] += 1
        
        report += "\nISSUES BY FILE:\n"
        report += "-" * 20 + "\n"
        for file_path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"{count:2d} issues: {file_path}\n"
        
        return report
    
    def check_dataset_availability(self) -> Dict[str, bool]:
        """Check if real datasets are available."""
        datasets_dir = self.base_dir / "datasets"
        if not datasets_dir.exists():
            return {}
        
        expected_datasets = {
            "crows-pairs": "data/crows_pairs_anonymized.csv",
            "winobias": "wino/data/anti_stereotyped_type1.txt.dev", 
            "winogender": "data/templates.tsv",
            "bbq": "BBQ.csv",
            "stereoset": "data/bias-bench/stereoset/test.json",
            "bold": "prompts.json",
            "truthfulqa": "TruthfulQA.csv"
        }
        
        availability = {}
        for dataset, expected_file in expected_datasets.items():
            dataset_path = datasets_dir / dataset
            file_path = dataset_path / expected_file
            availability[dataset] = dataset_path.exists() and (not expected_file or file_path.exists())
        
        return availability
    
    def run_analysis(self) -> None:
        """Run complete fake data analysis."""
        print("🔍 ALGOVERSE FAKE DATA ANALYSIS")
        print("=" * 40)
        
        # Scan codebase
        self.scan_directory(self.base_dir / "unified_pipeline")
        if (self.base_dir / "data_science").exists():
            self.scan_directory(self.base_dir / "data_science")
        
        # Generate report
        report = self.generate_report()
        print(report)
        
        # Check dataset availability
        print("\n📊 DATASET AVAILABILITY:")
        print("-" * 30)
        dataset_status = self.check_dataset_availability()
        available_count = sum(dataset_status.values())
        total_count = len(dataset_status)
        
        for dataset, available in dataset_status.items():
            status = "✅ Available" if available else "❌ Missing"
            print(f"{dataset:15s}: {status}")
        
        print(f"\nDatasets ready: {available_count}/{total_count}")
        
        # Save report to file
        report_path = self.base_dir / "FAKE_DATA_SCAN_RESULTS.md"
        with open(report_path, 'w') as f:
            f.write(f"# Fake Data Scan Results\n\n")
            f.write(f"**Scan Date**: {os.popen('date').read().strip()}\n\n")
            f.write(report)
            
            f.write(f"\n## Dataset Availability\n\n")
            for dataset, available in dataset_status.items():
                status = "✅" if available else "❌"
                f.write(f"- {dataset}: {status}\n")
        
        print(f"\n📄 Report saved to: {report_path}")
        
        # Return status
        critical_issues = len([i for i in self.issues_found if i['severity'] == 'CRITICAL'])
        if critical_issues > 0:
            print(f"\n🚨 CRITICAL: {critical_issues} critical fake data issues found!")
            return False
        elif self.issues_found:
            print(f"\n⚠️  WARNING: {len(self.issues_found)} potential issues found")
            return True
        else:
            print("\n✅ No fake data patterns detected!")
            return True

def main():
    """Main entry point."""
    detector = FakeDataDetector()
    success = detector.run_analysis()
    
    if not success:
        print("\n❌ Scan failed - critical fake data issues detected")
        exit(1)
    else:
        print("\n✅ Scan completed successfully")
        exit(0)

if __name__ == "__main__":
    main()