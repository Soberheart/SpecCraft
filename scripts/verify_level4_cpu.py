"""Save reproducible CPU acceptance evidence; synthetic examples are not experiments."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from level4.artifacts import sha256,write_json,read_json,runtime_environment,runtime_files,utc_now

ROOT=Path(__file__).resolve().parent.parent


def verify(output,probe_ssh=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    commands=[]
    def run(name,argv,expected=0,timeout=60):
        started=utc_now();clock=time.monotonic()
        try:
            result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
            stdout,stderr,code=result.stdout,result.stderr,result.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            stdout,stderr,code="",str(exc),None
        (output/f"{name}.stdout.txt").write_text(stdout,encoding="utf-8")
        (output/f"{name}.stderr.txt").write_text(stderr,encoding="utf-8")
        commands.append({"name":name,"argv":list(map(str,argv)),"cwd":str(ROOT),"started_utc":started,
            "ended_utc":utc_now(),"wall_s":time.monotonic()-clock,"exit_code":code,
            "expected_exit_code":expected,"expectation_met":code==expected if expected is not None else None})
        return code
    run("git_head",["git","rev-parse","HEAD"])
    run("git_status",["git","status","--short"])
    run("git_diff",["git","diff","--binary"])
    run("level4_tests",[sys.executable,"-m","unittest","discover","-s","tests","-p","test_level4_*.py","-v"])
    run("all_tests",[sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py","-v"])
    cli=ROOT/"scripts/level4.py"
    run("cli_help",[sys.executable,str(cli),"--help"])
    for name in ("audit_level4_representation","summarize_level4_development"):
        run(name+"_help",[sys.executable,str(ROOT/f"scripts/{name}.py"),"--help"])
    run("pool_config",[sys.executable,str(cli),"validate-pool-config","--config","configs/level4/development_pool.json"])
    for family in ("A","B"):
        run("uncalibrated_"+family,[sys.executable,str(cli),"validate-spec","--spec",f"configs/level4/task_{family}_uncalibrated.json","--experiment"],2)
    sources=list((ROOT/"level4").glob("*.py"))+list((ROOT/"tests").glob("test_level4_*.py"))
    sources += [ROOT/f"scripts/{name}.py" for name in ("level4","audit_level4_representation","summarize_level4_development","verify_level4_cpu")]
    for path in sources:ast.parse(path.read_text(encoding="utf-8"),filename=str(path),feature_version=(3,9))
    for path in (ROOT/"configs/level4").glob("*.json"):read_json(path)
    pbs=(ROOT/"scripts/run_level4_pool.pbs").read_bytes()
    if b"\r" in pbs or not pbs.startswith(b"#!/bin/bash\n"):
        raise ValueError("PBS script must use LF with a valid Bash shebang")
    write_json(output/"syntax_and_json.json",{"status":"passed","parser_feature_version":"3.9",
        "pbs_lf_encoding":"passed; Bash syntax remains unrun on this host",
        "python_files":[str(p.relative_to(ROOT)) for p in sources],"json_files":[p.name for p in (ROOT/"configs/level4").glob("*.json")],
        "limitation":"Parsing is not execution on Python 3.9; JSON Schema package validation not available"})
    # Saved example exercises the actual CLI and file boundary, with explicit synthetic labels.
    sys.path.insert(0,str(ROOT/"tests"))
    from test_level4_pipeline import fixture
    example=output/"synthetic_example";example.mkdir()
    pool,_,spec=fixture(example)
    run("synthetic_evaluate",[sys.executable,str(cli),"evaluate","--spec",str(spec),"--pool",str(pool),
        "--trajectory",str(pool/"predictions/candidate_000/trajectory.npz"),"--output",str(example/"evaluation.json"),"--prediction","--synthetic-only"])
    run("synthetic_freeze",[sys.executable,str(cli),"freeze-selection","--spec",str(spec),"--pool",str(pool),
        "--output",str(example/"selection.json"),"--synthetic-only"])
    write_json(example/"WARNING.json",{"kind":"synthetic_CPU_contract_example","material_experiment":False,
        "research_evidence":False,"thresholds":"analytic fixture only; cannot authorize simulation"})
    assets={}
    for path in (ROOT/"sim/mpm/assets").rglob("*"):
        if path.is_file():assets[path.relative_to(ROOT).as_posix()]={"sha256":sha256(path),"bytes":path.stat().st_size}
    provenance=read_json(ROOT/"sim/mpm/assets/meshes/processed/board-board.provenance.json")
    asset_match={key:assets[provenance[key]]["sha256"]==provenance[key+"_sha256"] for key in ("source_sdf","output_obj")}
    inventory=[]
    for directory,subdirs,files in os.walk(ROOT):
        subdirs[:]=[d for d in subdirs if d not in (".git","__pycache__",".venv","venv","node_modules") and Path(directory,d)!=output]
        for name in files:
            if name.endswith(".pth") or name in ("log.pkl","observations.npz") or name.startswith("comparison_frame"):
                path=Path(directory,name);inventory.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size})
    environment={**runtime_environment(),"interpreter":sys.executable,
        "available_modules":{name:importlib.util.find_spec(name) is not None for name in ("numpy","torch","scipy","omegaconf","yaml","jsonschema","matplotlib","taichi")},
        "executables":{name:shutil.which(name) for name in ("python","bash","ssh","nvidia-smi","git")}}
    write_json(output/"environment.json",environment)
    write_json(output/"inventory.json",{"board_assets":assets,"board_provenance_hashes_match":asset_match,
        "experiment_inputs_found":inventory,"server_inputs":"not accessed by CPU inventory"})
    if probe_ssh:
        run("ssh_probe",["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","NSCC-ASPIRE2A","pwd"],expected=None,timeout=20)
    write_json(output/"commands.json",commands)
    write_json(output/"tested_source_hashes.json",{p.relative_to(ROOT).as_posix():sha256(p) for p in runtime_files(ROOT)})
    passed=all(row["expectation_met"] is not False for row in commands) and all(asset_match.values())
    write_json(output/"result.json",{"status":"passed" if passed else "failed","created_utc":utc_now(),
        "scope":"CPU contracts and legacy helper regressions; no GPU prediction or physical simulation",
        "ssh_probe_is_acceptance_test":False,"pbs_bash_syntax":"not run: usable Bash not located on this Windows host",
        "commands_sha256":sha256(output/"commands.json"),"source_hashes_sha256":sha256(output/"tested_source_hashes.json")})
    print(json.dumps({"status":"passed" if passed else "failed","output":str(output)},ensure_ascii=False))
    return 0 if passed else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--probe-ssh",action="store_true",help="Read-only BatchMode NSCC access check")
    args=parser.parse_args();sys.exit(verify(args.output,args.probe_ssh))
