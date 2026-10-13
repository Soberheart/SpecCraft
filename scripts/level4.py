"""CPU evaluator and guarded fixed-pool CLI. See docs/level4_protocol.md."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from level4.artifacts import read_json,write_json,load_grid,load_trajectory
from level4.evaluate import evaluate
from level4.spec import validate_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    check=commands.add_parser("validate-spec")
    check.add_argument("--spec",type=Path,required=True)
    check.add_argument("--experiment",action="store_true")
    pool_check=commands.add_parser("validate-pool-config")
    pool_check.add_argument("--config",type=Path,required=True)
    evaluate_parser=commands.add_parser("evaluate")
    evaluate_parser.add_argument("--spec",type=Path,required=True)
    evaluate_parser.add_argument("--pool",type=Path,required=True)
    evaluate_parser.add_argument("--trajectory",type=Path,required=True)
    evaluate_parser.add_argument("--output",type=Path,required=True)
    evaluate_parser.add_argument("--prediction",action="store_true")
    evaluate_parser.add_argument("--synthetic-only",action="store_true",help="CPU fixtures only; cannot execute material experiments")
    prepare_parser=commands.add_parser("prepare")
    prepare_parser.add_argument("--dyn-artifacts",type=Path,required=True)
    prepare_parser.add_argument("--config",type=Path,required=True)
    prepare_parser.add_argument("--output",type=Path,required=True)
    freeze=commands.add_parser("freeze-selection")
    freeze.add_argument("--pool",type=Path,required=True)
    freeze.add_argument("--spec",type=Path,required=True)
    freeze.add_argument("--output",type=Path,required=True)
    freeze.add_argument("--synthetic-only",action="store_true")
    snapshot=commands.add_parser("snapshot")
    snapshot.add_argument("--output",type=Path,required=True)
    reset=commands.add_parser("audit-reset")
    reset.add_argument("--pools",type=Path,nargs="+",required=True)
    reset.add_argument("--output",type=Path,required=True)
    gate=commands.add_parser("validate-reset-gate")
    gate.add_argument("--reset-audit",type=Path,required=True)
    gate.add_argument("--config",type=Path,required=True)
    gate.add_argument("--dyn-artifacts",type=Path,required=True)
    for stage in ("predict","execute","analyze"):
        sub=commands.add_parser(stage)
        sub.add_argument("--pool",type=Path,required=True)
        if stage=="predict":
            sub.add_argument("--chunk-size",type=int,default=4096)
        if stage=="analyze":
            sub.add_argument("--spec",type=Path,required=True)
            sub.add_argument("--output",type=Path,required=True)
        if stage in ("execute","analyze"):
            sub.add_argument("--selection",type=Path,help="Prospective prediction-only manifest")
        if stage=="execute":
            sub.add_argument("--candidate-id",action="append",help="Development reset probes only")
            sub.add_argument("--all-candidates",action="store_true",help="Execute full diagnostic pool under frozen selection")
    args=parser.parse_args()
    if args.command=="validate-spec":
        validate_spec(read_json(args.spec),experiment=args.experiment)
        result={"status":"valid","experiment":args.experiment}
    elif args.command=="validate-pool-config":
        from level4.pipeline import validate_pool_config
        validate_pool_config(read_json(args.config));result={"status":"valid_development_config"}
    elif args.command=="evaluate":
        result=evaluate(read_json(args.spec),load_trajectory(args.trajectory,load_grid(args.pool)),prediction=args.prediction,experiment=not args.synthetic_only)
        write_json(args.output,result)
    else:
        from level4.pipeline import prepare,predict,execute,analyze,freeze_selection
        if args.command=="prepare":
            result=prepare(args.dyn_artifacts,args.output,args.config)
        elif args.command=="predict":
            result=predict(args.pool,args.chunk_size)
        elif args.command=="execute":
            if args.selection and args.candidate_id:
                parser.error("Do not override frozen selections with candidate IDs")
            result=execute(args.pool,args.selection,args.candidate_id,args.all_candidates)
        elif args.command=="freeze-selection":
            result=freeze_selection(args.pool,args.spec,args.output,experiment=not args.synthetic_only)
        elif args.command=="snapshot":
            from level4.artifacts import snapshot_runtime
            result={"status":"snapshot_saved","sha256":snapshot_runtime(Path(__file__).resolve().parent.parent,args.output)}
        elif args.command=="audit-reset":
            from level4.pipeline import audit_reset
            result=audit_reset(args.pools,args.output)
        elif args.command=="validate-reset-gate":
            from level4.pipeline import verify_reset_gate
            result=verify_reset_gate(args.reset_audit,args.config,args.dyn_artifacts)
        else:
            result=analyze(args.pool,args.spec,args.output,args.selection)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    try:
        main()
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        print(json.dumps({"status":"invalid_input","error":str(exc)},ensure_ascii=False),file=sys.stderr)
        sys.exit(2)
