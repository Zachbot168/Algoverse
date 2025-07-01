import os
import json
import argparse
import torch
import transformers
from datetime import datetime

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.benchmark.seat import SEATRunner
from bias_bench.util import generate_experiment_id


class BERTBiasTester:
    """
    Comprehensive bias testing suite for BERT variants.
    Tests baseline models across StereoSet, CrowS-Pairs, and SEAT benchmarks.
    """
    
    def __init__(self, persistent_dir="./results", data_dir="./data"):
        self.persistent_dir = persistent_dir
        self.data_dir = data_dir
        self.results = {}
        
        # Model configurations
        self.models_to_test = {
            "bert-base-uncased": {
                "model_class": "BertForMaskedLM", 
                "model_path": "bert-base-uncased",
                "description": "BERT Base - main encoder baseline"
            },
            "roberta-base": {
                "model_class": "RobertaForMaskedLM",
                "model_path": "roberta-base", 
                "description": "RoBERTa Base - strong comparative variant"
            },
            "google/electra-small-discriminator": {
                "model_class": "ElectraForMaskedLM",
                "model_path": "google/electra-small-discriminator",
                "description": "ELECTRA Small - efficient, diverse architecture"
            },
            "bert-large-uncased": {
                "model_class": "BertForMaskedLM",
                "model_path": "bert-large-uncased", 
                "description": "BERT Large - scaled baseline"
            }
        }
        
        # Bias types to test
        self.bias_types = ["gender", "race", "religion"]  # These are the main ones with data
        
        # SEAT tests to run
        self.seat_tests = [
            "sent-weat1", "sent-weat2", "sent-weat3", "sent-weat4",
            "sent-weat5", "sent-weat6", "sent-weat7", "sent-weat8"
        ]
        
        # Create output directories
        os.makedirs(f"{self.persistent_dir}/stereoset", exist_ok=True)
        os.makedirs(f"{self.persistent_dir}/crows", exist_ok=True)
        os.makedirs(f"{self.persistent_dir}/seat", exist_ok=True)
    
    def sanitize_model_name(self, model_name):
        """Convert model name to a safe filename by replacing slashes with underscores."""
        return model_name.replace("/", "_")
        
    def load_model_and_tokenizer(self, model_name, model_class, model_path):
        """Load model and tokenizer for testing."""
        print(f"Loading {model_name} ({model_class})...")
        
        try:
            # Load the appropriate model class
            if model_class == "ElectraForMaskedLM":
                # ELECTRA uses a different model class
                model = transformers.ElectraForMaskedLM.from_pretrained(model_path)
            else:
                model = getattr(transformers, model_class).from_pretrained(model_path)
            
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None
    
    def run_stereoset_test(self, model_name, model, tokenizer, model_path):
        """Run StereoSet bias evaluation."""
        print(f"\n--- Running StereoSet for {model_name} ---")
        
        results = {}
        
        for bias_type in self.bias_types:
            print(f"Testing {bias_type} bias...")
            
            try:
                # Create StereoSet runner
                runner = StereoSetRunner(
                    intrasentence_model=model,
                    tokenizer=tokenizer,
                    input_file=f"{self.data_dir}/stereoset/test.json",
                    model_name_or_path=model_path,
                    batch_size=1,
                    is_generative=False,
                    bias_type=bias_type
                )
                
                # Run evaluation
                bias_results = runner()
                results[bias_type] = bias_results
                
                print(f"  {bias_type} bias test completed")
                
            except Exception as e:
                print(f"  Error in {bias_type} bias test: {e}")
                results[bias_type] = {"error": str(e)}
        
        # Save results with sanitized filename
        safe_model_name = self.sanitize_model_name(model_name)
        output_file = f"{self.persistent_dir}/stereoset/{safe_model_name}_stereoset.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_crows_test(self, model_name, model, tokenizer, model_path):
        """Run CrowS-Pairs bias evaluation."""
        print(f"\n--- Running CrowS-Pairs for {model_name} ---")
        
        results = {}
        
        for bias_type in self.bias_types:
            print(f"Testing {bias_type} bias...")
            
            try:
                # Use the actual CrowS-Pairs dataset file
                input_file = f"{self.data_dir}/crows/crows_pairs_anonymized.csv"
                
                # Create CrowS-Pairs runner
                runner = CrowSPairsRunner(
                    model=model,
                    tokenizer=tokenizer,
                    input_file=input_file,
                    is_generative=False,
                    bias_type=bias_type
                )
                
                # Run evaluation
                bias_results = runner()
                results[bias_type] = bias_results
                
                print(f"  {bias_type} bias test completed")
                
            except Exception as e:
                print(f"  Error in {bias_type} bias test: {e}")
                results[bias_type] = {"error": str(e)}
        
        # Save results with sanitized filename
        safe_model_name = self.sanitize_model_name(model_name)
        output_file = f"{self.persistent_dir}/crows/{safe_model_name}_crows.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_seat_test(self, model_name, model, tokenizer, model_path):
        """Run SEAT bias evaluation."""
        print(f"\n--- Running SEAT for {model_name} ---")
        
        results = {}
        
        try:
            # Create SEAT runner
            runner = SEATRunner(
                model=model,
                tokenizer=tokenizer,
                tests=self.seat_tests,
                data_dir=f"{self.data_dir}/seat",
                n_samples=100000,
                parametric=False,
                experiment_id=f"seat_{model_name}"
            )
            
            # Run evaluation
            seat_results = runner()
            results = seat_results
            
            print(f"  SEAT tests completed")
            
        except Exception as e:
            print(f"  Error in SEAT tests: {e}")
            results = {"error": str(e)}
        
        # Save results with sanitized filename
        safe_model_name = self.sanitize_model_name(model_name)
        output_file = f"{self.persistent_dir}/seat/{safe_model_name}_seat.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_full_evaluation(self, tests_to_run=["stereoset", "crows", "seat"]):
        """Run complete bias evaluation suite."""
        print("="*60)
        print("BERT BIAS TESTING SUITE")
        print("="*60)
        print(f"Testing {len(self.models_to_test)} models across {len(tests_to_run)} benchmark(s)")
        print(f"Bias types: {', '.join(self.bias_types)}")
        print(f"Results will be saved to: {self.persistent_dir}")
        print("="*60)
        
        for model_name, config in self.models_to_test.items():
            print(f"\n{'='*20} {model_name.upper()} {'='*20}")
            print(f"Description: {config['description']}")
            
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer(
                model_name, config["model_class"], config["model_path"]
            )
            
            if model is None or tokenizer is None:
                print(f"Skipping {model_name} due to loading error")
                continue
            
            # Initialize results for this model
            self.results[model_name] = {
                "model_info": config,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run selected tests
            if "stereoset" in tests_to_run:
                stereoset_results = self.run_stereoset_test(
                    model_name, model, tokenizer, config["model_path"]
                )
                self.results[model_name]["stereoset"] = stereoset_results
            
            if "crows" in tests_to_run:
                crows_results = self.run_crows_test(
                    model_name, model, tokenizer, config["model_path"]
                )
                self.results[model_name]["crows"] = crows_results
            
            if "seat" in tests_to_run:
                seat_results = self.run_seat_test(
                    model_name, model, tokenizer, config["model_path"]
                )
                self.results[model_name]["seat"] = seat_results
            
            # Clear memory
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"Completed testing for {model_name}")
        
        # Save comprehensive results
        self.save_comprehensive_results()
        self.generate_summary_report()
        
        return self.results
    
    def save_comprehensive_results(self):
        """Save all results to a comprehensive file."""
        output_file = f"{self.persistent_dir}/bert_bias_comprehensive_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nComprehensive results saved to: {output_file}")
    
    def generate_summary_report(self):
        """Generate a human-readable summary report."""
        report_file = f"{self.persistent_dir}/bert_bias_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("BERT BIAS TESTING SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Tested: {len(self.results)}\n")
            f.write(f"Bias Types: {', '.join(self.bias_types)}\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{'-'*30}\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write(f"Description: {results['model_info']['description']}\n")
                f.write(f"{'-'*30}\n")
                
                # StereoSet summary
                if "stereoset" in results:
                    f.write("\nStereoSet Results:\n")
                    for bias_type, stereo_results in results["stereoset"].items():
                        if "error" not in stereo_results:
                            f.write(f"  {bias_type}: Available\n")
                        else:
                            f.write(f"  {bias_type}: Error - {stereo_results['error']}\n")
                
                # CrowS-Pairs summary
                if "crows" in results:
                    f.write("\nCrowS-Pairs Results:\n")
                    for bias_type, crows_results in results["crows"].items():
                        if "error" not in crows_results:
                            f.write(f"  {bias_type}: Available\n")
                        else:
                            f.write(f"  {bias_type}: Error - {crows_results['error']}\n")
                
                # SEAT summary
                if "seat" in results:
                    f.write(f"\nSEAT Results: {'Available' if 'error' not in results['seat'] else 'Error'}\n")
            
            f.write(f"\n{'='*50}\n")
            f.write("For detailed results, see individual JSON files and the comprehensive results file.\n")
        
        print(f"Summary report saved to: {report_file}")


def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description="BERT Bias Testing Suite")
    parser.add_argument(
        "--persistent_dir", 
        type=str, 
        default="./bias_test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--data_dir",
        type=str, 
        default="./data",
        help="Directory containing bias test datasets"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["stereoset", "crows", "seat"],
        default=["stereoset", "crows", "seat"],
        help="Which bias tests to run"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert-base-uncased", "roberta-base", "google/electra-small-discriminator", "bert-large-uncased"],
        default=None,
        help="Specific models to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = BERTBiasTester(
        persistent_dir=args.persistent_dir,
        data_dir=args.data_dir
    )
    
    # Filter models if specified
    if args.models:
        original_models = tester.models_to_test.copy()
        tester.models_to_test = {
            k: v for k, v in original_models.items() 
            if k in args.models
        }
    
    # Run evaluation
    results = tester.run_full_evaluation(tests_to_run=args.tests)
    
    print("\n" + "="*60)
    print("BIAS TESTING COMPLETED")
    print("="*60)
    print(f"Results saved to: {args.persistent_dir}")
    print(f"Models tested: {list(results.keys())}")
    print(f"Tests run: {args.tests}")


if __name__ == "__main__":
    main()