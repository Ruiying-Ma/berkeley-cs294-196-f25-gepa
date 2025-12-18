import os
import time
import numpy as np
import tempfile
import subprocess
import pickle
import json

M_TRACE_TIMEOUT = {
  "alibaba_300.oracleGeneral.bin": 20,
  "alibaba_353.oracleGeneral.bin": 19,
  "alibaba_419.oracleGeneral.bin": 18,
  "alibaba_448.oracleGeneral.bin": 11,
  "alibaba_526.oracleGeneral.bin": 25,
  "alibaba_684.oracleGeneral.bin": 21,
  "alibaba_758.oracleGeneral.bin": 1,
  "alibaba_802.oracleGeneral.bin": 3,
  "alibaba_805.oracleGeneral.bin": 1,
  "alibaba_816.oracleGeneral.bin": 1,
  "alibaba_818.oracleGeneral.bin": 1,
  "alibaba_820.oracleGeneral.bin": 1,
  "alibaba_821.oracleGeneral.bin": 14,
  "alibaba_822.oracleGeneral.bin": 13,
  "alibaba_849.oracleGeneral.bin": 16,
  "alibaba_860.oracleGeneral.bin": 19,
  "alibaba_863.oracleGeneral.bin": 17,
  "alibaba_868.oracleGeneral.bin": 15,
  "alibaba_879.oracleGeneral.bin": 16,
  "alibaba_890.oracleGeneral.bin": 1,
  "alibaba_892.oracleGeneral.bin": 1,
  "alibaba_914.oracleGeneral.bin": 29,
  "alibaba_979.oracleGeneral.bin": 2,
  "alibaba_980.oracleGeneral.bin": 4,
  "tencent_17315.oracleGeneral.bin": 53,
  "tencent_19195.oracleGeneral.bin": 1,
  "tencent_22680.oracleGeneral.bin": 1,
  "tencent_2408.oracleGeneral.bin": 1,
  "tencent_24971.oracleGeneral.bin": 1,
  "tencent_25070.oracleGeneral.bin": 1,
  "tencent_25073.oracleGeneral.bin": 1,
  "tencent_25100.oracleGeneral.bin": 6,
  "tencent_25103.oracleGeneral.bin": 6,
  "tencent_25115.oracleGeneral.bin": 1,
  "tencent_25136.oracleGeneral.bin": 1,
  "tencent_25149.oracleGeneral.bin": 1,
  "tencent_25155.oracleGeneral.bin": 6,
  "tencent_25183.oracleGeneral.bin": 2,
  "tencent_25204.oracleGeneral.bin": 2,
  "tencent_25226.oracleGeneral.bin": 1,
  "tencent_25243.oracleGeneral.bin": 1,
  "tencent_25269.oracleGeneral.bin": 1,
  "tencent_25294.oracleGeneral.bin": 1,
  "tencent_25307.oracleGeneral.bin": 1,
  "tencent_25316.oracleGeneral.bin": 2,
  "tencent_25330.oracleGeneral.bin": 1,
  "tencent_25499.oracleGeneral.bin": 1,
  "tencent_26011.oracleGeneral.bin": 1,
}

def _log_metrics(code: str, metrics: dict):
    dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openevolve_metrics.jsonl")
    log_entry = {
        "code": code,
        "metrics": metrics
    }
    with open(dst_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def run_with_timeout(trace_path, timeout_seconds=20):
    # Create a tempfile to store cache_simuate result
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".pickle", delete=False) as temp_file:
        result_path = temp_file.name
    
    try:
        process = subprocess.Popen(
            ['python', os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_simulate.py"), "--trace_path", str(trace_path), "--result_path", result_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            start = time.time()
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            end = time.time()
            exit_code = process.returncode
            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")
            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")
            # Load the results
            if os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                results[f"run_time"] = round(end - start, 4)
                return results
            else:
                raise RuntimeError("Results file not found")
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")
    finally:
        # Ensure process is killed if still running
        if process and process.poll() is None:
            print("Killing subprocess...")
            process.kill()
            process.wait()
        if os.path.exists(result_path):
            os.remove(result_path)

def evaluate(code_str: str, trace_path: str):
    try: 
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "My.py"), 'w') as f:
            f.write(code_str)
        try:
            timeout_seconds = M_TRACE_TIMEOUT.get(os.path.basename(trace_path), max(M_TRACE_TIMEOUT.values()))
            result_dict = run_with_timeout(trace_path, timeout_seconds=timeout_seconds) #TIMEOUT for stage2 (init: ~100s)
            assert os.path.basename(trace_path) in result_dict, "Miss workload result."
            assert len(result_dict) == 2, "Miss workload result."
            assert isinstance(result_dict[os.path.basename(trace_path)], float), "Invalid result type"
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "My.py"), 'w') as f:
                f.write("")
            _log_metrics(code_str, result_dict)
            assert "runs_successfully" not in result_dict, "runs_successfully error"
            result_dict["runs_successfully"] = 1.0
            return result_dict
        except TimeoutError:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "My.py"), 'w') as f:
                f.write("")
            return {
                "runs_successfully": 0.0,
                'error': f"Error - Timeout"
            }
        except Exception as e:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "My.py"), 'w') as f:
                f.write("")
            return {
                "runs_successfully": 0.0,
                'error': f"Error - {str(e)}"
            }
            
    except Exception as e:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "My.py"), 'w') as f:
            f.write("")
        return {
            "runs_successfully": 0.0,
            'error': f"Error - {str(e)}"
        }